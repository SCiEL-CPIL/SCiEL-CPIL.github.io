import time
import random

# EVOLVE-BLOCK-START
def construct():
    # Precompute popcount for 17-bit integers
    # Max value 2^17 - 1
    popcount_table = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)
        
    def popcount(x):
        return popcount_table[x]

    n = 17
    m = n * (n - 1) // 2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    start_time = time.time()
    time_limit = 115.0
    
    # State
    colors = [random.randint(0, 3) for _ in range(m)]
    adj = [[0] * n for _ in range(4)]
    
    def update_adj(c, u, v, add):
        if add:
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        else:
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            
    for idx, (u, v) in enumerate(edges):
        c = colors[idx]
        update_adj(c, u, v, True)
        
    def count_c6_with_edge(c, u, v):
        mask_v = adj[c][v] & ~(1 << u)
        mask_u = adj[c][u] & ~(1 << v)
        
        count = 0
        # Iterate w in mask_v
        w = 0
        while w < n:
            if mask_v & (1 << w):
                Nw = adj[c][w] & ~(1 << u) & ~(1 << v)
                # Iterate z in mask_u
                z = 0
                while z < n:
                    if mask_u & (1 << z):
                        if w != z:
                            Nz = adj[c][z] & ~(1 << u) & ~(1 << v)
                            Nw_excl = Nw & ~(1 << z)
                            Nz_excl = Nz & ~(1 << w)
                            
                            # Count edges between Nw_excl and Nz_excl
                            # Sum over x in Nw_excl of popcount(adj[c][x] & Nz_excl)
                            x = 0
                            while x < n:
                                if Nw_excl & (1 << x):
                                    count += popcount(adj[c][x] & Nz_excl)
                                x += 1
                    z += 1
            w += 1
        return count

    total_c6 = 0
    for idx, (u, v) in enumerate(edges):
        total_c6 += count_c6_with_edge(colors[idx], u, v)
        
    best_colors = colors[:]
    best_score = total_c6
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, m - 1)
        u, v = edges[idx]
        c_old = colors[idx]
        
        best_delta = 0
        best_c_new = -1
        
        for c_new in range(4):
            if c_new == c_old:
                continue
            
            delta_remove = count_c6_with_edge(c_old, u, v)
            delta_add = count_c6_with_edge(c_new, u, v)
            delta = delta_add - delta_remove
            
            if delta < best_delta:
                best_delta = delta
                best_c_new = c_new
                
        if best_c_new != -1 and best_delta <= 0:
            update_adj(c_old, u, v, False)
            update_adj(best_c_new, u, v, True)
            colors[idx] = best_c_new
            total_c6 += best_delta
            
            if total_c6 < best_score:
                best_score = total_c6
                best_colors = colors[:]
                
            if best_score == 0:
                break
                
    if best_score == 0:
        return {'n': n, 'colors': best_colors}
    else:
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
# EVOLVE-BLOCK-END
