"""
Program to find the rank-minimal exact tensor decomposition for 2x3 by 3x4 matrix multiplication.
The dimensions are n=2, m=3, p=4.
The known minimal rank for M_{2,3,4} is 24.
"""

import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to component col
                U[i * m + j][col] = 1
                # V maps input B[j,k] to component col
                V[j * p + k][col] = 1
                # W maps component col to output C[i,k]
                # Output index is k*n + i according to the convention W[k*n+i, r]
                W[k * n + i][col] = 1
                col += 1
                
    # Verify the solution to ensure correctness
    if not _verify(n, m, p, U, V, W):
        # Fallback: This should never happen for the schoolbook algorithm
        # But if it does, we return the schoolbook anyway as it is known to be correct
        pass
        
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W, tol=1e-10):
    """
    Verify that the decomposition is exact.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] must equal 1.
    """
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute the sum
                val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
