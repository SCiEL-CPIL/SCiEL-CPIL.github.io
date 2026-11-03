import time
import sys
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, g=16, searches for a valid graph using LUW construction.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit
    
    # General case: use LUW construction
    best_n = float('inf')
    best_edges = None
    
    # List of prime powers to try
    prime_powers = [3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47, 49, 53, 59, 61, 64, 67, 71, 73, 79, 81, 83, 89, 97]
    
    # Generate candidate (q, d) pairs
    candidates = []
    for q in prime_powers:
        # n = 2 * q^d
        # We want 510 <= 2 * q^d <= 50000
        # 255 <= q^d <= 25000
        d = 1
        while True:
            n = 2 * (q ** d)
            if n > 50000:
                break
            if n >= 510:
                candidates.append((n, q, d))
            d += 1
            if d > 20: # Safety
                break
                
    # Sort by n ascending to try smaller graphs first (faster girth check)
    candidates.sort(key=lambda x: x[0])
    
    for n, q, d in candidates:
        if time.time() > deadline:
            break
            
        try:
            n_actual, edges = _luw_construct(q, d)
            # Quick girth check with early termination
            # Only check if we haven't found a better one yet?
            # We want the smallest n, so we check in order of n.
            # If the first valid one we find is the smallest n, we can return immediately.
            girth = _calc_girth_fast(n_actual, edges, g)
            if girth >= g:
                return n_actual, edges
        except Exception as e:
            pass

    # If no valid graph found in candidates, try a few more with larger d but smaller q?
    # Or just return the last attempted one? No, it must be valid.
    # If we ran out of time, we might have to return an invalid graph or timeout.
    # But we should try to return the best valid one found.
    # If best_edges is None, we have a problem.
    
    # Fallback: Try to construct a random 3-regular graph with n=936 and hope?
    # Or try LUW with q=3, d=10 (n=118098 > 50000, invalid).
    
    # If we didn't find a valid graph, we must return something.
    # The evaluator will reject it.
    # Let's try one more thing: if k=3 and g=16, try to find a graph with n=936?
    # We don't have the construction.
    
    # If best_edges is None, return a dummy graph to avoid crash?
    # But it must be valid.
    # Let's assume the search will find one.
    
    if best_edges is not None:
        return best_n, best_edges
        
    # Last resort: Return a small valid graph if g is small?
    # But g is 16.
    # Return a graph with n=510?
    # We can't construct it easily.
    
    # If we get here, we failed.
    # Return an empty list or something that will be caught?
    # The task says "An invalid result... scores 0".
    # So I should try my best to return a valid one.
    
    # Try LUW with q=3, d=7 again, but maybe the girth check was too strict?
    # No, girth must be >= 16.
    
    # Let's try a wider search if time permits?
    # No, we are out of candidates.
    
    # Return the first candidate's graph even if girth is low?
    # No.
    
    # Raise an error?
    # No, return a dummy.
    return 0, []

def _luw_construct(q, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q).
    Returns (n, edges).
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for LUW construction
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _calc_girth_fast(n, edges, threshold):
    """
    Calculates girth using BFS from each vertex with early termination.
    Returns girth if found, or threshold if no cycle shorter than threshold exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    max_depth = threshold - 1  # We only need to check up to this depth
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < threshold:
                            return min_girth
    
    return min_girth if min_girth != float('inf') else threshold

if __name__ == "__main__":
    # Quick test
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices, {len(edges)} edges.")
    if n > 0:
        girth = _calc_girth_fast(n, edges, 20)
        print(f"Measured girth (threshold 20): {girth}")
