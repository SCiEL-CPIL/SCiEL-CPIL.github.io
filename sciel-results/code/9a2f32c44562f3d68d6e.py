"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # We construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={3,4,5}.
    # Edges in base: (i, j+3) for all i in 0..2, j in 0..2.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex (side, base, shift) where side in {0,1}, base in 0..2, shift in 0..m-1.
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m) where v[i][j] is the voltage.
    
    # For girth g, we need no cycles shorter than g.
    # The Moore bound for 3-regular graph with girth g gives lower bound on n.
    # For g=16, Moore bound is 512. The best known is 936.
    # With K_{3,3} lift, n = 6*m, so m = n/6.
    # To get n close to 512, m would be ~85.33, so m=86 gives n=516.
    # But we need to find voltages that give girth >= g.
    
    # Strategy: try random voltage assignments for various m values,
    # check girth, and return the smallest m that works.
    
    # For efficiency, we'll try m from a reasonable lower bound upward.
    # Lower bound: Moore bound for (3, g) cage.
    # For g even, Moore bound n >= 3 * (2^(g/2 - 1) - 1)
    # For g=16: n >= 3 * (2^7 - 1) = 3 * 127 = 381
    # So m >= ceil(381/6) = 64
    
    # Let's compute Moore bound properly
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        # For odd girth, the bound is different
        # n >= (3/2) * (2^(g-1)//2 - 1) approximately
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # We'll try m from m_min upward, but to keep within time limits,
    # we'll try a limited number of m values and random voltage assignments.
    
    def check_girth(m, voltages, required_girth):
        """Check if the lifted graph has girth >= required_girth.
        voltages: 3x3 matrix where voltages[i][j] is the voltage on edge from A_i to B_j.
        Returns True if girth >= required_girth, False otherwise.
        """
        n = 6 * m
        # Build adjacency list
        # Vertex encoding: side * 3 * m + base * m + shift
        # side 0 = A, side 1 = B
        adj = [[] for _ in range(n)]
        
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = 0 * 3 * m + i * m + s
                    w = 1 * 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
        
        # BFS from each vertex to find shortest cycle
        # Actually, for girth checking, we can do BFS from each vertex and track
        # the first time we revisit a vertex (excluding the parent).
        # But that's O(n * (n + e)) which is too slow for large n.
        
        # Better approach: BFS from each vertex, stop when depth reaches required_girth/2
        # If we find a back edge at depth < required_girth/2, girth is too small.
        
        half_girth = required_girth // 2
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        
        return True
    
    # Try random voltage assignments for increasing m
    random.seed(42)
    
    # For each m, try several random voltage assignments
    num_trials = 50
    
    for m in range(m_min, m_min + 200):  # Try up to 200 values of m
        if 6 * m > 2000:  # Cap to avoid too large graphs
            break
        
        for trial in range(num_trials):
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth(m, voltages, g):
                # Build the graph
                n = 6 * m
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = voltages[i][j]
                        for s in range(m):
                            u = 0 * 3 * m + i * m + s
                            w = 1 * 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
        
        # If no random assignment worked for this m, try a few more
        # Actually, let's also try some structured assignments
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
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
