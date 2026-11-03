"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
    # Convert to lists of lists as required by the interface?
    # The interface says: factor matrices of shapes ... with integer/half-integer entries.
    # The seed program returns lists of lists.
    # I should return lists of lists to be safe, or numpy arrays if allowed.
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank)..."
    # The seed uses lists. I will use lists to match the seed's type exactly, 
    # although numpy is often accepted. 
    # Let's check the seed: U = [[0] * rank ...]
    # I will convert back to lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
