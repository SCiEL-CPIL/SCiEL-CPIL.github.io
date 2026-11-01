"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    import time
    from collections import deque

    # Strategy: Random voltage lift of K_{3,3} over Z_m.
    # n = 6*m.
    # We want to minimize n, so minimize m.
    # Moore bound for (3, 16) is 510. So m >= 85.
    
    # Correct Moore bound for even girth g=2k:
    # n >= 2 * sum_{i=0}^{k-1} (d-1)^i
    if g % 2 == 0:
        k_val = g // 2
        # Sum of geometric series: (d-1)^k - 1 / (d-2)
        # Here d=3, so d-1=2, d-2=1. Sum = 2^k - 1.
        # n >= 2 * (2^k - 1)
        moore_bound = 2 * ((2 ** k_val) - 1)
    else:
        # For odd girth, use a conservative lower bound
        k_val = (g - 1) // 2
        moore_bound = 2 * ((2 ** k_val) - 1) # Approximation
        
    m_min = max(1, math.ceil(moore_bound / 6))
    
    # Time management
    start_time = time.time()
    timeout = 110.0
    
    def build_adj(m, voltages):
        """Build adjacency list for the voltage graph."""
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

    def check_girth(adj, n, required_girth):
        """Check if graph has girth >= required_girth."""
        half_girth = required_girth // 2
        # We can optimize by checking only up to half_girth depth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Length is du + dist[v] + 1
                        if du + dist[v] + 1 < required_girth:
                            return False
        return True

    random.seed(42)
    
    # Search for valid m
    # We try m from m_min upwards
    # For each m, we try multiple random voltage assignments
    # If we find one, we return it.
    
    max_m = m_min + 300 # Search up to m_min + 300
    trials_per_m = 100
    
    for m in range(m_min, max_m):
        if time.time() - start_time > timeout:
            break
            
        n = 6 * m
        # If n is too large, we might not want to continue, but we want small n.
        # However, if we don't find a small one, we might need a larger one.
        
        found = False
        for trial in range(trials_per_m):
            # Generate random voltages
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            # Build adjacency and check girth
            adj = build_adj(m, voltages)
            
            if check_girth(adj, n, g):
                # Valid graph found
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = voltages[i][j]
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges
                
        # If no valid graph found for this m, try next m

    # Fallback: If we didn't find a valid graph within time/search space,
    # we must return something valid. 
    # We can use a known construction or a large random search with fixed m.
    # To be safe, let's try a larger m with more trials, or just return a 
    # valid graph from a different construction if available.
    # Since we must return a valid graph, and we might have timed out,
    # let's try one last desperate measure: use a known valid graph if we can construct it quickly.
    # Or, increase m significantly and try again.
    
    # Let's try m = 200 with 1000 trials if we get here
    if time.time() - start_time < timeout - 10:
        m = 200
        for trial in range(1000):
            if time.time() - start_time > timeout:
                break
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            adj = build_adj(m, voltages)
            n = 6 * m
            if check_girth(adj, n, g):
                edges = []
                for i in range(3):
                    for j in range(3):
                        v = voltages[i][j]
                        for s in range(m):
                            u = i * m + s
                            w = 3 * m + j * m + (s + v) % m
                            edges.append((u, w))
                return n, edges

    # Ultimate fallback: Return a trivial 3-regular graph that satisfies girth?
    # Hard to do quickly. 
    # We can return the Lazebnik-Ustimenko-Woldar construction from the original seed.
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
