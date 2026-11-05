import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the exact tensor decomposition for matrix multiplication.
    For 3x4 * 4x7, the bilinear complexity (tensor rank) of the standard
    schoolbook algorithm is 3*4*7 = 84.
    
    The tensor T is defined by C_ik = sum_j A_ij B_jk.
    We decompose T into a sum of rank-1 tensors:
    T = sum_{i,j,k} e_{ik} (W) \otimes e_{ij} (U) \otimes e_{jk} (V)
    
    Note: The interface requires U, V, W matrices such that:
    T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
    where U corresponds to A indices, V to B indices, W to C indices.
    
    Specifically:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    For each term r corresponding to (i, j, k):
    U[i*m + j, r] = 1
    V[j*p + k, r] = 1
    W[k*n + i, r] = 1
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i, col j -> i*m + j
                # Index in V: row j, col k -> j*p + k
                # Index in W: row k, col i -> k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
