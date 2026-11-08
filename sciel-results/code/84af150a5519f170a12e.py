import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the indices for the standard rank-18 decomposition
    # Each term corresponds to a unique (i, j, k) triple
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
                
    rank = len(indices)
    
    # Initialize matrices as zero arrays
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Populate matrices based on the algebraic terms
    # The tensor is T_{ijk} = U_{ij}^r V_{jk}^r W_{ki}^r
    # For the standard decomposition, each term r corresponds to a specific (i, j, k)
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    for r, (i, j, k) in enumerate(indices):
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
        
    # Convert to lists to match the required interface if strict types are needed, 
    # though numpy arrays are often accepted. The prompt says "integer/half-integer entries".
    # The seed returns lists of lists. I will return lists of lists to be safe.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
