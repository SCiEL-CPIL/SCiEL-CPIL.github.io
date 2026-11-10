import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Moore bound for (k,g): 2*(k-1)*((k-1)^(g/2-1) - 1)/(k-2) for even g
    # For k=4, g=10: 2*3*(3^4 - 1)/2 = 3*80 = 240, but actual lower bound is 243
    if k == 4 and g == 10:
        moore_bound = 243
    else:
        # General Moore bound
        if g % 2 == 0:
            r = g // 2 - 1
            moore_bound = 2 * (k - 1) * ((k - 1) ** r - 1) // (k - 2) if k > 2 else 2
        else:
            r = (g - 1) // 2
            moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    best_n = None
    best_edges = None
    
    # First, try the LUW construction which is known to be optimal for many cages
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        if _check_girth(n_luw, edges_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
    
    # Try to find a smaller graph via random search if we have time
    if time.time() < deadline - 5:
        # Try to construct graphs with increasing n
        for n in range(moore_bound, (best_n if best_n else 400)):
            if time.time() > deadline:
                break
            
            # Try multiple random constructions for this n
            attempts = 0
            max_attempts = 20  # Reduced since random is unlikely to beat LUW
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                edges = _try_random_4regular(n, k)
                if edges is not None:
                    if _check_girth(n, edges, k, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_edges = edges
                        break  # Found a valid graph for this n, move to next n
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: use LUW construction
    return _luw_construct(k, g)


def _try_random_4regular(n, k):
    """Try to construct a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    
    # Use configuration model
    half_edges = n * k // 2
    stubs = [i for i in range(n) for _ in range(k)]
    random.shuffle(stubs)
    
    # Pair up stubs
    edges = []
    seen = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Self-loop
        edge = (min(u, v), max(u, v))
        if edge in seen:
            return None  # Multi-edge
        seen.add(edge)
        edges.append(edge)
    
    return edges


def _check_girth(n, edges, k, g):
    """Check if the graph has girth at least g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Non-backtracking BFS from each vertex to find shortest cycle
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # We track the shortest distance to each (u, v) state
        dist = {}
        queue = deque()
        
        # Initialize: from start, go to each neighbor
        for v in adj[start]:
            dist[(v, start)] = 1
            queue.append((v, start))
        
        while queue:
            u, prev = queue.popleft()
            d = dist[(u, prev)]
            
            if d >= g:
                # We've reached depth g without finding a shorter cycle
                # But we need to check if we can close a cycle
                continue
            
            for v in adj[u]:
                if v == prev:
                    continue  # Don't backtrack
                
                if v == start and d + 1 >= g:
                    # Found a cycle of length d+1 >= g, which is fine
                    continue
                
                if v == start:
                    # Found a cycle of length d+1 < g
                    return False
                
                state = (v, u)
                if state not in dist or dist[state] > d + 1:
                    dist[state] = d + 1
                    queue.append(state)
        
    return True


def _luw_construct(k, g):
    """LUW construction for cage graphs."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
