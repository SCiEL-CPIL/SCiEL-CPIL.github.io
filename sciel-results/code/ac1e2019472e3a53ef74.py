"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # Construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # Base: K_{3,3} with A={0,1,2}, B={3,4,5}, edges (i,j+3) for i,j in 0..2.
    # Voltage v[i][j] in Z_m on edge A_i - B_j.
    # Lifted graph: 6m vertices, vertex (side, base, shift), side in {0,1}.
    # Edge: (0,i,s) -- (1,j,(s+v[i][j])%m)
    
    # For girth g=16, need no cycles of length 4,6,8,10,12,14.
    # Moore bound for (3,16): n >= 3*(2^7-1) = 381, so m >= 64.
    # Best known: 936 vertices. We aim for m as small as possible.
    
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    def build_adj(m, voltages):
        """Build adjacency list for the lifted graph."""
        n = 6 * m
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
        return adj
    
    def check_girth_fast(m, voltages, required_girth):
        """Efficient girth check using BFS with cycle detection.
        Returns True if girth >= required_girth."""
        n = 6 * m
        adj = build_adj(m, voltages)
        half_g = required_girth // 2
        
        for start in range(n):
            # BFS from start, track distance and parent
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found: length = dist[u] + dist[v] + 1
                        if dist[u] + dist[v] + 1 < required_girth:
                            return False
        return True
    
    def build_edges(m, voltages):
        """Build edge list for the lifted graph."""
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    edges.append((u, w))
        return n, edges
    
    # Strategy: Try m from m_min upward with random voltage assignments.
    # Use a time budget to ensure we finish within 120s.
    import time
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    random.seed(42)
    
    # Number of trials per m value
    trials_per_m = 100
    
    for m in range(m_min, m_min + 300):
        if time.time() - start_time > time_budget:
            break
        if 6 * m > 1200:  # Cap at 1200 vertices
            break
        
        for trial in range(trials_per_m):
            if time.time() - start_time > time_budget:
                break
            
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth_fast(m, voltages, g):
                return build_edges(m, voltages)
    
    # Fallback: use Lazebnik-Ustimenko-Woldar construction
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
# EVOLVE-BLOCK-END
