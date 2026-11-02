# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    # Split p=8 into 3, 3, 2
    splits = [3, 3, 2]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    k_offset = 0
    for p_sub in splits:
        # Generate schoolbook terms for n x m x p_sub
        for i in range(n):
            for j in range(m):
                for k_sub in range(p_sub):
                    k = k_offset + k_sub
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
        k_offset += p_sub
        
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r_idx in range(n * m):
            U[r_idx][c] = U_cols[c][r_idx]
        for r_idx in range(m * p):
            V[r_idx][c] = V_cols[c][r_idx]
        for r_idx in range(p * n):
            W[r_idx][c] = W_cols[c][r_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
