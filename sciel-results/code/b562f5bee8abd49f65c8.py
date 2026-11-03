import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to list of lists to match the expected interface if strict, 
    # or return numpy arrays if the evaluator accepts them. 
    # The seed returned lists of lists. I will return lists of lists to be safe,
    # but constructed via numpy for a distinct code path.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
