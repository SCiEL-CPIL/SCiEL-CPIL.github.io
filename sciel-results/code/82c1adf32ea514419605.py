import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy:
    # 1. Quick attempt at random lifts of Petersen (if time permits, low effort)
    # 2. Deterministic LUW construction with optimized depth search
    
    best_n = float('inf')
    best_edges = None
    
    # 1. Try random lifts of Petersen graph (low effort, high potential reward)
    # We only do this if we have plenty of time.
    if time.time() < deadline - 30:
        petersen_edges = _get_petersen_edges()
        # Try M from 52 to 60 (n=520 to 600)
        for M in range(52, 61):
            if time.time() > deadline - 30:
                break
            n = 10 * M
            if n >= best_n:
                break
            # Try 5 random lifts
            for _ in range(5):
                if time.time() > deadline - 30:
                    break
                edges = _random_lift(petersen_edges, M)
                if edges is not None and _has_girth_at_least(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
            if best_n == n:
                break
        
        if best_n != float('inf'):
            return best_n, best_edges

    # 2. Deterministic LUW Construction
    # Find the smallest d such that the LUW graph has girth >= g
    # Girth of LUW graph with depth d is approximately 2d + 2 or 2d.
    # We will iterate d and check.
    
    # Start d from a reasonable lower bound
    # Moore bound implies d must be large enough.
    # For k=3, n ~ 3^d.
    # If g=16, d=7 gives n~2187.
    
    min_d = max(3, g // 2 - 2)
    max_d = g + 5  # Safety cap
    
    for d in range(min_d, max_d + 1):
        if time.time() > deadline - 5:
            break
            
        try:
            n_luw, edges_luw = _construct_luw_optimized(k, d)
        except Exception:
            continue
            
        if n_luw < best_n:
            # Verify girth
            if _has_girth_at_least(n_luw, edges_luw, g):
                best_n = n_luw
                best_edges = edges_luw
                # If we found a valid graph, we can stop if we are confident
                # that smaller d would have failed (which we checked in previous iterations)
                # Since we are iterating d upwards, the first valid one is the smallest n for this family.
                # However, different d might yield different n.
                # For LUW, n increases with d. So first valid d gives smallest n.
                break

    if best_n == float('inf'):
        # Fallback: Return a valid but potentially large graph
        # Use a larger d to ensure validity
        n_luw, edges_luw = _construct_luw_optimized(k, max_d)
        return n_luw, edges_luw

    return best_n, best_edges

def _get_petersen_edges():
    edges = []
    for i in range(5):
        edges.append((i, (i+1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i+1) % 5]))
    for i in range(5):
        edges.append((i, i+5))
    return edges

def _random_lift(base_edges, M):
    n_base = 10
    n_lift = n_base * M
    perm_map = {}
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        if key not in perm_map:
            p = list(range(M))
            random.shuffle(p)
            perm_map[key] = p
    lift_edges = []
    for u, v in base_edges:
        key = (min(u, v), max(u, v))
        p = perm_map[key]
        for j in range(M):
            u_node = u * M + j
            v_node = v * M + p[j]
            lift_edges.append((u_node, v_node))
    return lift_edges

def _has_girth_at_least(n, edges, g):
    """Check girth >= g using shortest path from each neighbor back to vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        for neighbor in adj[start]:
            # BFS from neighbor to start, not using edge (start, neighbor)
            dist = [-1] * n
            dist[neighbor] = 0
            q = deque([neighbor])
            found = False
            while q:
                u = q.popleft()
                if u == start:
                    found = True
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        # Don't use the edge (start, neighbor) in reverse
                        if u == neighbor and v == start:
                            continue
                        dist[v] = dist[u] + 1
                        q.append(v)
            if found:
                cycle_len = dist[start] + 1
                if cycle_len < g:
                    return False
    return True

def _construct_luw_optimized(k, d):
    """
    Deterministic LUW construction for k-regular graphs.
    Optimized for k=3.
    """
    if k != 3:
        # Fallback to generic LUW if k != 3
        return _construct_luw_generic(k, d)
        
    # For k=3, we use the specific LUW rules.
    # q = 3
    q = 3
    # The rule for the tree unfolding
    # Side 0: (x, y) -> (y, x + y) ? No, specific to LUW.
    # We reuse the logic from the seed but cleaned up.
    
    # The seed's rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _construct_luw_generic(k, d):
    # Fallback for other k
    q = k
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q

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
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
