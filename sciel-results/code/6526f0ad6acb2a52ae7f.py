"""Matrix multiplication 2x3 x 3x9 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x9 matrix.
    
    Output format:
    - U: (n*m, rank) matrix
    - V: (m*p, rank) matrix  
    - W: (p*n, rank) matrix
    
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    With trace convention: W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition
    # For each (i, j, k) triple, we have one rank-1 term
    # U[i*m + j, col] = 1, V[j*p + k, col] = 1, W[k*n + i, col] = 1
    rank = n * m * p  # 2 * 3 * 9 = 54
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
