import numpy as np
import time
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 5, 6
    
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strassen 2x2x2 terms
    strassen = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,0):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    def build_decomp(i_groups, j_groups, k_groups):
        """Build decomposition given groupings. Returns (U, V, W, rank) or None."""
        total_rank = 0
        blocks = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        blocks.append(('s', ig, jg, kg, 7))
                        total_rank += 7
                    else:
                        r = si * sj * sk
                        blocks.append(('b', ig, jg, kg, r))
                        total_rank += r
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        col = 0
        
        for btype, ig, jg, kg, r in blocks:
            if btype == 's':
                for u_pat, v_pat, w_pat in strassen:
                    for (li, lj), val in u_pat.items():
                        U[ig[li]*m + jg[lj]][col] = val
                    for (lj, lk), val in v_pat.items():
                        V[jg[lj]*p + kg[lk]][col] = val
                    for (lk, li), val in w_pat.items():
                        W[kg[lk]*n + ig[li]][col] = val
                    col += 1
            else:
                for li in range(len(ig)):
                    for lj in range(len(jg)):
                        for lk in range(len(kg)):
                            U[ig[li]*m + jg[lj]][col] = 1
                            V[jg[lj]*p + kg[lk]][col] = 1
                            W[kg[lk]*n + ig[li]][col] = 1
                            col += 1
        
        if col != total_rank:
            return None
        if not verify(U, V, W, total_rank):
            return None
        return U, V, W, total_rank
    
    # Try multiple partitionings
    best = None
    best_rank = n * m * p + 1
    
    partitionings = [
        # Key: 2x2x2 Strassen blocks with [0,1]x[0,1]x[0,1] pattern
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        # Alternative: different i partition
        ([[0],[1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        # Another: group i as [0,1,2]
        ([[0,1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        # Fine-grained
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1,2],[3,4,5]]),
        ([[0,1],[2]], [[0,1,2],[3,4]], [[0,1],[2,3,4,5]]),
        # More variations
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0],[1,2]], [[0,1],[2,3],[4]], [[0,1],[2,3],[4,5]]),
        ([[0,1],[2]], [[0,1],[2,3],[4]], [[0,1,2],[3,4,5]]),
    ]
    
    for ig, jg, kg in partitionings:
        if time.time() > deadline:
            break
        result = build_decomp(ig, jg, kg)
        if result is not None:
            _, _, _, r = result
            if r < best_rank:
                best_rank = r
                best = result
        elif time.time() > deadline:
            break
    
    if best is not None:
        U, V, W, rank = best
        return {"U": U, "V": V, "W": W}
    
    # Direct construction: partition [0,1],[2] x [0,1],[2,3],[4] x [0,1],[2,3],[4,5]
    # This should give rank 72
    ig = [[0,1],[2]]
    jg = [[0,1],[2,3],[4]]
    kg = [[0,1],[2,3],[4,5]]
    result = build_decomp(ig, jg, kg)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            return {"U": U, "V": V, "W": W}
    
    # Fallback
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}
