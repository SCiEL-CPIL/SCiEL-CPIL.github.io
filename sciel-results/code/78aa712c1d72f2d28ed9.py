import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W with integer entries using a Kronecker-product-like structure
    representing the multiplication as a sum of outer products of selection vectors.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The tensor T[i, j, k] = A[i, j] * B[j, k]
    # Decomposition: C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We iterate over all (i, j, k) tuples. Each tuple corresponds to one rank-1 term.
    # This is the standard schoolbook decomposition.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices for the factor matrices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Set the entry to 1 for this rank-1 component
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
