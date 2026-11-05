"""Matrix multiplication 3x5x7 - Rank minimal decomposition attempt."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Currently uses the schoolbook decomposition which has rank 105.
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize U, V, W as numpy arrays for efficient construction
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
                
    # Convert to lists to match the expected output format if strict typing is required
    # The prompt implies integer/half-integer entries, lists of lists are safe
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
