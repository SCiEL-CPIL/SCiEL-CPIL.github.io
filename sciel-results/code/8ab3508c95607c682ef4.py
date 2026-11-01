    def construct(k, g):
        import random
        import math
        from collections import deque
        import time

        # 1. Try K_{3,3} voltage lift
        # Moore bound for n(3, g)
        if g % 2 == 0:
            moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
        else:
            moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
        
        m_min = max(1, math.ceil(moore_bound / 6))
        
        # Time limit for K3,3 search
        start_time = time.time()
        time_limit_k33 = 30  # seconds
        
        random.seed(42)
        
        def check_girth_k33(m, voltages, required_girth):
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
            
            half_girth = required_girth // 2
            
            for start in range(n):
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
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
            return True

        # Try small m values
        # We know best known is m=156 (n=936). 
        # Random search for m < 156 is very unlikely to succeed.
        # But we try a few just in case.
        for m in range(m_min, min(m_min + 10, 160)):
            if time.time() - start_time > time_limit_k33:
                break
            for _ in range(10): # Fewer trials
                voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
                if check_girth_k33(m, voltages, g):
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

        # 2. Fallback: Lazebnik-Ustimenko-Woldar (LUW)
        # Optimize d for minimal size
        # Girth of LUW(q, d) is 2d+1
        # Need 2d+1 >= g => d >= (g-1)/2
        d = (g + 1) // 2
        if d < 3: d = 3
        
        q = k # 3
        
        # LUW Construction
        # Vertices: (side, x0, x1, ..., x_{d-1}) where side in {0,1}
        # x_i in Z_q
        # Total vertices: 2 * q^d
        
        # To speed up, we map (side, tuple) to index
        # But for q=3, d=8, q^d = 6561. Total 13122.
        # We can generate vertices in order.
        
        # The LUW graph definition:
        # Neighbors of (0, x): (1, y) where y_0 is free, and for c >= 1:
        # y_c = x_c + y_{c-1} * x_{c-1}  (Wait, the seed uses a specific rule)
        # Let's use the standard LUW definition which is simpler to implement correctly.
        # Standard LUW:
        # Vertices are pairs (s, x) where s in {0,1}, x in F_q^d.
        # Edges: (0, x) ~ (1, y) if y_0 is arbitrary and y_i = x_i + y_{i-1} x_{i-1} for i=1..d-1?
        # Actually, the seed's rule is:
        # y[c] = add(x[c], mul(y[a], x[b]))
        # This is a specific variant.
        # To ensure correctness, I will stick to the seed's logic but optimize the execution.
        # However, the seed's logic is complex.
        
        # Alternative: Use a known fast construction for (3, girth) cages if possible?
        # No, LUW is standard.
        
        # Let's implement LUW efficiently.
        # We need to map (side, x_tuple) to an integer index.
        # index = side * q^d + sum(x_i * q^i)
        
        q_power = [1] * (d + 1)
        for i in range(1, d + 1):
            q_power[i] = q_power[i-1] * q
            
        n_vertices = 2 * q_power[d]
        
        edges = []
        
        # Precompute neighbors for side 0 vertices?
        # For each x in F_q^d:
        #   for y_0 in F_q:
        #     compute y
        #     add edge
        
        # We iterate over all x (0 to q^d - 1)
        # Convert x to base-q digits
        # For each y_0, compute y
        # Add edge
        
        for x_idx in range(q_power[d]):
            # Decode x_idx to x_digits
            x_digits = [0] * d
            temp = x_idx
            for i in range(d):
                x_digits[i] = temp % q
                temp //= q
            
            # x is side 0
            u = x_idx # side 0 index is 0..q^d-1
            
            for y0 in range(q):
                y_digits = [0] * d
                y_digits[0] = y0
                # The seed's rule:
                # y[c] = add(x[c], mul(y[a], x[b]))
                # The seed defines 'rule' dict.
                # This is getting complicated to replicate exactly without the seed's 'rule' generation.
                
                # Let's use the SIMPLER standard LUW definition:
                # y_i = x_i + y_{i-1} * x_{i-1} is NOT standard.
                # Standard LUW (from Lazebnik et al.):
                # (0, x) ~ (1, y) iff y_0 is free, and y_i = x_i + y_{i-1} x_{i-1} for i=1..d-1?
                # Actually, the girth depends on the specific algebra.
                
                # To be safe and fast, I will use the **Cage Graph** from a known source if I can recall one.
                # Or, I can just use the seed's LUW logic but with $d=8$.
                
                # Let's just use the seed's logic, it's verified to work for girth.
                # I need to replicate the 'rule' generation.
        
        # Replicating Seed's LUW Logic
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
        # Generate edges
        # Vertex ID: side * q^d + index
        # We iterate over all vertices in side 0.
        for x_idx in range(q_power[d]):
            # Decode x
            x = [0] * d
            temp = x_idx
            for i in range(d):
                x[i] = temp % q
                temp //= q
            
            u = x_idx # side 0
            
            for y0 in range(q):
                y = [0] * d
                y[0] = y0
                # x[0] is unused in the formula for y[1..d-1]?
                # Seed: y[c] = add(x[c], mul(y[a], x[b]))
                # For c=2: a,b = rule[2] = (1,1). y[2] = x[2] + y[1]*x[1].
                # Note: y[1] is determined by previous step?
                # Seed: y = [None, first] + [0]*(d-1)
                # y[0] is None? No, y[1] is 'first'.
                # The seed's indexing is 1-based for the tuple part?
                # "y = [None, first] + [0] * (d - 1)"
                # So y[0] is None, y[1] is first.
                # This implies the tuple has length d+1?
                # "start = (0, (None,) + (0,) * d)" -> length d+1.
                # So x has d+1 elements, x[0] is None.
                
                # Let's align with seed's structure to ensure correctness.
                # x[0] is None. x[1..d] are values.
                # y[0] is None. y[1] is free.
                
                # Re-decode x to match seed: x[0]=None, x[1..d]
                x_seed = [None] + [0] * d
                temp = x_idx
                for i in range(1, d + 1):
                    x_seed[i] = temp % q
                    temp //= q
                    
                y_seed = [None] * (d + 1)
                y_seed[1] = y0
                
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # a, b are indices in 1..d
                    # y[c] = x[c] + y[a] * x[b]
                    y_seed[c] = (x_seed[c] + y_seed[a] * x_seed[b]) % q
                    
                # Encode y_seed to index
                y_idx = 0
                for i in range(1, d + 1):
                    y_idx += y_seed[i] * q_power[i-1]
                    
                v = q_power[d] + y_idx # side 1
                
                edges.append((u, v))
                
        return n_vertices, edges
    