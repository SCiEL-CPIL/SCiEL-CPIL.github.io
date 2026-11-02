"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18
    
    terms = []
    
    # Schoolbook decomposition
    # C[i, k] = sum_j A[i, j] B[j, k]
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][c] = val
        for idx, val in v.items():
            V[idx][c] = val
        for idx, val in w.items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
