import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-30 tensor decomposition for 2x3 * 3x5 matrix multiplication.
    Uses a Kronecker-product structure derived from identity matrices.
    The decomposition is C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    where the sum is over c, and for each c, the indices (i,j,k) are fixed.
    This corresponds to the schoolbook algorithm with rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Generate indices for the Kronecker structure
    # i in [0, n), j in [0, m), k in [0, p)
    # The "Kronecker" aspect is that we iterate through all combinations.
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
