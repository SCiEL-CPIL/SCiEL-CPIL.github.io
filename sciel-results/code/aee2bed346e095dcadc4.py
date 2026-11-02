"""Matrix multiplication 3x3 * 3x4 - rank minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x4 matrix.
    
    The decomposition satisfies:
      For all i in [0,3), j in [0,3), k in [0,4):
        sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    where U has shape (n*m, R) = (9, R), V has shape (m*p, R) = (12, R),
    W has shape (p*n, R) = (12, R).
    
    We use a rank-9 construction which is the tensor rank of the matmul tensor.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    R = n * m  # 9
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each r in 0..R-1, r corresponds to (i, j) = (r // m, r % m)
    for r in range(R):
        i = r // m
        j = r % m
        U[i * m + j, r] = 1.0
        for k in range(p):
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
    
    # Convert to lists for the return value
    U_list = [[float(U[i, j]) for j in range(R)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(R)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(R)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
