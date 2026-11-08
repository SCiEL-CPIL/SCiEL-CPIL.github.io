"""Program to find a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists of lists as required by the interface if strict, 
    # but the prompt says "integer/half-integer entries". 
    # The seed program returns lists of lists. 
    # I will return lists of lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
