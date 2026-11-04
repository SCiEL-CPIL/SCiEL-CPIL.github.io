import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, the known minimal rank is 30 (schoolbook).
    We construct U, V, W with integer entries.
    
    The decomposition is based on the standard basis vectors for the row, column, and output indices.
    The tensor is the sum of n*m*p rank-1 terms.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We use numpy for efficient construction, then convert to lists
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Generate indices for all (i, j, k) tuples
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We can use np.indices or just loops. For small N, loops are fine and clear.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Row indices in the factor matrices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Set the 1s
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    # Convert to lists to match the interface of the seed program
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
