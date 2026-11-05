import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    # --- Helper: GF(16) multiplication for the Clebsch graph ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def solve_limited(N, time_limit, seed_colors=None):
        t_start = time.time()
        t_end = t_start + time_limit
        
        if seed_colors and len(seed_colors) == N * (N - 1) // 2:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1

        M2 = [M[c] @ M[c] for c in range(4)]
        
        # Precompute list of edges (i, j, idx) for faster iteration
        edges = []
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                edges.append((i, j, idx))
                idx += 1

        def get_bad_edges():
            bad = []
            for i, j, idx_e in edges:
                c = C[i][j]
                if M2[c][i][j] > 0:
                    bad.append((i, j, c, idx_e))
            return bad

        bad_edges = get_bad_edges()
        
        if not bad_edges:
            return colors.tolist()

        current_T = 5.0
        cooling = 0.99
        min_T = 0.1
        
        # Precompute common neighbors for faster delta calculation?
        # Delta for changing (u,v) from c_old to c_new:
        # Triangles lost in c_old: common neighbors of u,v in c_old
        # Triangles gained in c_new: common neighbors of u,v in c_new
        
        while time.time() < t_end:
            if not bad_edges:
                return colors.tolist()
            
            u, v, c_old, idx_u_v = random.choice(bad_edges)
            
            # Calculate delta for each new color
            best_delta = float('inf')
            best_c = -1
            
            # Number of common neighbors in color c for edge (u,v) is M2[c][u][v]
            # But M2[c][u][v] counts paths u-k-v.
            # If we change (u,v) to color c_new:
            # We remove edge from c_old.
            # Any triangle u-v-k in c_old is broken if (u,v) is the only edge connecting them? No.
            # A triangle u-v-k in c_old exists if M[c_old][u,v]=1, M[c_old][v,k]=1, M[c_old][k,u]=1.
            # If we change (u,v) to c_new, this triangle is no longer monochromatic c_old.
            # So we lose M2[c_old][u,v] triangles of color c_old?
            # Wait. M2[c][u][v] = sum_k M[c][u,k]M[c][k,v].
            # If M[c][u,v]=1, then for any k such that M[c][u,k]=1 and M[c][k,v]=1,
            # the triple (u,v,k) is a triangle of color c.
            # So yes, the number of monochromatic triangles containing edge (u,v) of color c is M2[c][u,v].
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                # Triangles removed from c_old
                # This is exactly M2[c_old][u,v]
                # Triangles added to c_new
                # This is exactly M2[c_new][u,v]
                # BUT: M2[c_new][u,v] counts paths u-k-v in color c_new.
                # If we set (u,v) to c_new, do we create new triangles?
                # Yes, for any k such that M[c_new][u,k]=1 and M[c_new][k,v]=1,
                # the triple (u,v,k) becomes a triangle of color c_new.
                # So delta = M2[c_new][u,v] - M2[c_old][u,v]
                
                k_old = M2[c_old][u][v]
                k_new = M2[c_new][u][v]
                delta = k_new - k_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                C[u][v] = best_c
                C[v][u] = best_c
                colors[idx_u_v] = best_c
                
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                # Efficient update of M2?
                # M2[c] = M[c] @ M[c]
                # Updating M[c] at (u,v) and (v,u) changes M2[c] in rows/cols u and v.
                # Specifically, M2[c][i,j] changes only if i or j is u or v?
                # M2[c][i,j] = sum_k M[c][i,k]M[c][k,j].
                # If we change M[c][u,v], it affects terms where k=u or k=v.
                # So M2[c][i,j] changes for all i,j if k=u or k=v is involved?
                # No, M2[c][i,j] is a scalar.
                # The change in M2[c] is:
                # dM2[c] = dM[c] @ M[c] + M[c] @ dM[c]
                # dM[c] is non-zero only at (u,v) and (v,u).
                # Let's just recompute M2[c_old] and M2[best_c].
                # N=50, matrix mult is 50^3 = 125k ops. Very fast.
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    seed_32 = build_seed_32()
    
    # Try N=50
    # We have ~115s. 
    # Each solve_limited(50, T) takes T seconds.
    # Let's try 5 restarts of 20s = 100s.
    # If we have time, try N=48, 46.
    
    for i in range(5):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(50, 20.0)
        if res:
            return {'n': 50, 'colors': res}
            
    for i in range(3):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(48, 15.0)
        if res:
            return {'n': 48, 'colors': res}
            
    for i in range(2):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(46, 10.0)
        if res:
            return {'n': 46, 'colors': res}
            
    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
