import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Construct the schoolbook decomposition for matrix multiplication n x m * m x p -> n x p
    # Rank is n * m * p
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    rank = n * m * p
    
    # Initialize with zeros
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Iterate over all combinations of i, j, k to set the rank-1 terms
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s for this rank-1 component
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    # Convert to lists of lists as per interface requirement (if strictly lists are expected)
    # The seed used lists. NumPy arrays might be accepted, but let's match the seed's type safety.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
