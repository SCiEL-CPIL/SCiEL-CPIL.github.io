"""
2x2 matrix multiplication rank-minimal exact tensor decomposition.

The tensor rank of 2x2 matrix multiplication is known to be 7 (Strassen, 1969).
This program constructs the Strassen decomposition directly and returns it.
No search is needed since the optimal rank is known.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    
    Uses Strassen's algorithm which achieves the known minimal rank of 7.
    
    The factor matrices follow the convention:
      U has shape (n*m, rank) = (4, 7), indexed U[i*m+j, r]
      V has shape (m*p, rank) = (4, 7), indexed V[j*p+k, r]  
      W has shape (p*n, rank) = (4, 7), indexed W[k*n+i, r]
    
    where A is (n x m), B is (m x p), C = A @ B is (n x p).
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Strassen's 7 products:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U[0*2+0][0] = 1   # A[0,0]
    U[1*2+1][0] = 1   # A[1,1]
    V[0*2+0][0] = 1   # B[0,0]
    V[1*2+1][0] = 1   # B[1,1]
    
    # M2: (A[1,0] + A[1,1]) * B[0,0]
    U[1*2+0][1] = 1   # A[1,0]
    U[1*2+1][1] = 1   # A[1,1]
    V[0*2+0][1] = 1   # B[0,0]
    
    # M3: A[0,0] * (B[0,1] - B[1,1])
    U[0*2+0][2] = 1   # A[0,0]
    V[0*2+1][2] = 1   # B[0,1]
    V[1*2+1][2] = -1  # B[1,1]
    
    # M4: A[1,1] * (B[1,0] - B[0,0])
    U[1*2+1][3] = 1   # A[1,1]
    V[1*2+0][3] = 1   # B[1,0]
    V[0*2+0][3] = -1  # B[0,0]
    
    # M5: (A[0,0] + A[0,1]) * B[1,1]
    U[0*2+0][4] = 1   # A[0,0]
    U[0*2+1][4] = 1   # A[0,1]
    V[1*2+1][4] = 1   # B[1,1]
    
    # M6: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[1*2+0][5] = 1   # A[1,0]
    U[0*2+0][5] = -1  # A[0,0]
    V[0*2+0][5] = 1   # B[0,0]
    V[0*2+1][5] = 1   # B[0,1]
    
    # M7: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[0*2+1][6] = 1   # A[0,1]
    U[1*2+1][6] = -1  # A[1,1]
    V[1*2+0][6] = 1   # B[1,0]
    V[1*2+1][6] = 1   # B[1,1]
    
    # W[k*n + i, r] — output reconstruction
    # C[0,0] = M1 + M4 - M5 + M7
    W[0*2+0][0] = 1   # M1
    W[0*2+0][3] = 1   # M4
    W[0*2+0][4] = -1  # M5
    W[0*2+0][6] = 1   # M7
    
    # C[0,1] = M3 + M5
    W[1*2+0][2] = 1   # M3
    W[1*2+0][4] = 1   # M5
    
    # C[1,0] = M2 + M4
    W[0*2+1][1] = 1   # M2
    W[0*2+1][3] = 1   # M4
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[1*2+1][0] = 1   # M1
    W[1*2+1][1] = -1  # M2
    W[1*2+1][2] = 1   # M3
    W[1*2+1][5] = 1   # M6
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
