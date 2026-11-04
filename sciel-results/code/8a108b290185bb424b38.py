import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Constructs the tensor decomposition for 3x3 matrix multiplied by 3x4 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill in the schoolbook decomposition
    # T[i, j, k] = 1 if C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor entry is non-zero for (i, j, k) such that:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, column] = 1.0
                V[v_idx, column] = 1.0
                W[w_idx, column] = 1.0
                
                column += 1
                
    # Convert to lists as required by the interface if strict, 
    # but the prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The seed used lists of lists. I will return lists to be safe and match the seed interface exactly.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
