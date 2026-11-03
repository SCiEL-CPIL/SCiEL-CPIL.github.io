import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill in the standard decomposition
    # Each column r corresponds to a triplet (i, j, k)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i, j) -> i*m + j
                U[i * m + j, col] = 1.0
                # V is indexed by (j, k) -> j*p + k
                V[j * p + k, col] = 1.0
                # W is indexed by (k, i) -> k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists to match the expected interface format if strict
    # The seed returned lists of lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
