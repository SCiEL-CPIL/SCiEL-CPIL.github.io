import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    # For k=3, g=16
    # Moore bound: 1 + 3*(1+2+4+8+16+32+64+128) = 766
    # Since bipartite, n must be even
    
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g)
    
    # Try to find a (3,16)-cage
    # Start from Moore bound, go up
    # n must be even (bipartite)
    
    for n in range(766, 1000, 2):  # Try up to 1000
        if n % 2 != 0:
            continue
        half = n // 2
        # Try to build a random 3-regular bipartite graph with girth check
        for attempt in range(100):  # More attempts
            edges = _random_3reg_bipartite(half)
            if edges is None:
                continue
            if _check_girth_fast(edges, n, g):
                return n, edges
    
    # Fallback: return a valid but possibly larger graph
    # Use LUW construction
    return _luw_construct(k, g)

def _random_3reg_bipartite(half):
    """Generate a random 3-regular bipartite graph with half vertices on each side."""
    # Side A: 0 to half-1
    # Side B: half to n-1
    n = half * 2
    
    # Use configuration model with rejection for multi-edges
    # Try multiple random pairings
    for _ in range(20):
        stubs_a = [i for i in range(half) for _ in range(3)]
        stubs_b = [i + half for i in range(half) for _ in range(3)]
        
        random.shuffle(stubs_a)
        random.shuffle(stubs_b)
        
        edges = set()
        valid = True
        for a, b in zip(stubs_a, stubs_b):
            if a > b:
                a, b = b, a
            if (a, b) in edges:
                valid = False
                break
            edges.add((a, b))
        
        if not valid:
            continue
        
        # Check all vertices have degree 3
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        
        return list(edges)
    
    return None

def _check_girth_fast(edges, n, g):
    """Fast girth check using BFS with early termination."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For girth g, we need to check for cycles of length < g
    # In bipartite graph, cycles are even: 2,4,6,...,g-2
    # So we only need to check up to depth (g-2)//2
    
    max_depth = (g - 2) // 2
    
    for start in range(n):
        # BFS up to max_depth
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent[node]:
                    continue
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                else:
                    # Found a cycle
                    cycle_len = dist[neighbor] + dist[node] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _luw_construct(k, g):
    """Fallback to LUW construction."""
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
