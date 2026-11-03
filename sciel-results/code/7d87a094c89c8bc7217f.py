"""Matrix multiplication 4x5x6."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals:
            vec[idx] = val
        return vec
    
    # Schoolbook: rank = n*m*p = 4*5*6 = 120
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_vec(n*m, [(i*m + j, 1)])
                v = make_vec(m*p, [(j*p + k, 1)])
                w = make_vec(p*n, [(k*n + i, 1)])
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in enumerate(U_cols[r]):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(V_cols[r]):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(W_cols[r]):
            if val != 0:
                W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
