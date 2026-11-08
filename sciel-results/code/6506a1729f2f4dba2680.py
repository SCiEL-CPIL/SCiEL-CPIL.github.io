"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # Direct multiplication decomposition
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # Term for each (i, j, k)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We will build the terms and then transpose to U, V, W
    # Each term is a tuple (u_vec, v_vec, w_vec)
    terms = []
    
    def make_vec(size, indices_vals):
        vec = [0] * size
        for idx, val in indices_vals.items():
            vec[idx] = val
        return vec

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                u_vec = make_vec(n*m, {u_idx: 1})
                v_vec = make_vec(m*p, {v_idx: 1})
                w_vec = make_vec(p*n, {w_idx: 1})
                terms.append((u_vec, v_vec, w_vec))
                
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n*m):
            U[i][c] = u[i]
        for i in range(m*p):
            V[i][c] = v[i]
        for i in range(p*n):
            W[i][c] = w[i]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
