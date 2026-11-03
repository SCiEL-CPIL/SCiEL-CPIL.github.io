"""Tensor decomposition for 3x3x7 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    # Dimensions: A is 3x3, B is 3x7, C is 3x7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W indexed as W[k*n + i, r] per trace convention
    
    n, m, p = 3, 3, 7
    total = n * m * p  # 63 terms for naive decomposition
    
    # Initialize factor matrices as lists of lists
    U = [[0.0] * total for _ in range(n * m)]
    V = [[0.0] * total for _ in range(m * p)]
    W = [[0.0] * total for _ in range(p * n)]
    
    # Each term corresponds to a unique (i, j, k) triple
    # U[i*m + j, t] = 1 for term t = i*m*p + j*p + k
    # V[j*p + k, t] = 1 for same term
    # W[k*n + i, t] = 1 for same term
    
    t = 0
    for i in range(n):
        base_u = i * m
        base_w_i = i
        for j in range(m):
            u_row = base_u + j
            base_v = j * p
            for k in range(p):
                U[u_row][t] = 1.0
                V[base_v + k][t] = 1.0
                W[k * n + base_w_i][t] = 1.0
                t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
