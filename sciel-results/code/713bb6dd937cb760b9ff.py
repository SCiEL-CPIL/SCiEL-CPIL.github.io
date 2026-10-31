"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # Voltage lift of K_{3,3} over Z_m for 3-regular graphs with girth >= g.
    # n = 6*m vertices.
    
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))

    def check_girth_efficient(m, voltages, required_girth):
        """Check girth using BFS from each vertex with early termination."""
        n = 6 * m
        half_girth = required_girth // 2
        
        # Precompute adjacency in compact form
        # Vertex: side*3*m + base*m + shift
        # For each vertex, its 3 neighbors
        
        def get_neighbors(v):
            side = v // (3 * m)
            rem = v % (3 * m)
            base = rem // m
            shift = rem % m
            if side == 0:
                # From A_base_shift to B_j_(shift + v[base][j])
                return [3 * m + j * m + (shift + voltages[base][j]) % m for j in range(3)]
            else:
                # From B_base_shift to A_i_(shift - v[i][base])
                return [i * m + (shift - voltages[i][base]) % m for i in range(3)]
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_girth:
                    continue
                for v in get_neighbors(u):
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    random.seed(42)
    
    # Try m values from m_min upward
    max_m = m_min + 150
    num_trials_per_m = 30
    
    for m in range(m_min, min(max_m, 200)):
        if 6 * m > 1500:
            break
        
        for trial in range(num_trials_per_m):
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth_efficient(m, voltages, g):
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
    
    # Fallback: Lazebnik-Ustimenko-Woldar
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
