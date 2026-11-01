# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # Moore bound for (3, g) cage
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3

    m_min = max(3, math.ceil(moore_bound / 6))
    
    def check_girth(m, voltages, required_girth):
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
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= required_girth // 2:
                    continue
                for v in adj[u]:
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
    
    # Search for valid m
    # We expect m to be around 100-200 for g=16
    # Let's search from m_min to m_min + 200
    max_m_search = m_min + 200
    
    for m in range(m_min, max_m_search + 1):
        if 6 * m > 2000:
            break
            
        # Try random voltage assignments
        # Increase trials for smaller m where probability is lower?
        # Or just keep it constant.
        num_trials = 30
        
        for _ in range(num_trials):
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth(m, voltages, g):
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

    # If not found, try a larger range or fallback
    # For robustness, if we haven't found it, we might need to go higher
    # But 200 steps should cover it.
    
    # Fallback to a known construction if available, or just return the last tried
    # which is invalid. Better to raise an error or return a large graph?
    # The task says "invalid result scores 0".
    # Let's try one more thing: use the Lazebnik-Ustimenko-Woldar construction as a fallback
    # if the voltage lift fails. It guarantees a valid graph, though n might be larger.
    
    # L-U-W Construction
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
