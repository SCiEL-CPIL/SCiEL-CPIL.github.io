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
    
    # Try to construct graphs with increasing n
    for n in range(moore_bound, 400):  # Cap at 400 for safety
        if time.time() > deadline:
            break
        
        # Try multiple random constructions for this n
        attempts = 0
        max_attempts = 50  # Adjust based on time
        
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
    """Check if the graph has girth at least g using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _luw_construct(k, g):
    """Fallback: LUW construction."""
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
