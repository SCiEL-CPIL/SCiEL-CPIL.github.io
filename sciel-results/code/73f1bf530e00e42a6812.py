"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the exact rank-54 tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The matrix multiplication tensor T has T[a, b, c] = 1 iff:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    for some i in [0, n), j in [0, m), k in [0, p).
    
    Here n=2, m=3, p=9.
    The schoolbook decomposition uses one term for each (i, j, k) triplet.
    Total rank = n * m * p = 2 * 3 * 9 = 54.
    
    For each term r = i * (m*p) + j * p + k:
    U[i*m + j, r] = 1
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1
    
    All other entries are 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the matrices
    for i in range(n):
        for j in range(m):
            a_idx = i * m + j
            for k in range(p):
                b_idx = j * p + k
                c_idx = k * n + i
                r = i * (m * p) + j * p + k
                
                U[a_idx][r] = 1
                V[b_idx][r] = 1
                W[c_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
