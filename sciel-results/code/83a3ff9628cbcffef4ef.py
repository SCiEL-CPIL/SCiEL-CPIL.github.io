"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2x2 matrix multiplication.
    
    Uses Strassen's algorithm which achieves rank 7 for 2x2 matrix multiplication.
    The output factor follows the official trace convention W[k*n+i, r].
    
    U has shape (n*m, rank) = (4, 7)
    V has shape (m*p, rank) = (4, 7)
    W has shape (p*n, rank) = (4, 7)
    
    For the decomposition to be valid:
    For each (i, j, k) in {0,1}^3:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # Then:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # U factors (row = i*m+j, col = r):
    # r=0: M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    # r=1: M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    # r=2: M3 = a(f-h)
    U[0, 2] = 1  # a
    # r=3: M4 = d(g-e)
    U[3, 3] = 1  # d
    # r=4: M5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    # r=5: M6 = (c-a)(e+f)
    U[2, 5] = 1  # c
    U[0, 5] = -1  # a
    # r=6: M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1  # d
    
    # V factors (row = j*p+k, col = r):
    # r=0: M1 = (a+d)(e+h)
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    # r=1: M2 = (c+d)e
    V[0, 1] = 1  # e
    # r=2: M3 = a(f-h)
    V[1, 2] = 1  # f
    V[3, 2] = -1  # h
    # r=3: M4 = d(g-e)
    V[2, 3] = 1  # g
    V[0, 3] = -1  # e
    # r=4: M5 = (a+b)h
    V[3, 4] = 1  # h
    # r=5: M6 = (c-a)(e+f)
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    # r=6: M7 = (b-d)(g+h)
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    
    # W factors (row = k*n+i, col = r):
    # C[0,0]: k=0,i=0 -> W[0*2+0] = W[0]
    # C[0,1]: k=1,i=0 -> W[1*2+0] = W[2]
    # C[1,0]: k=0,i=1 -> W[0*2+1] = W[1]
    # C[1,1]: k=1,i=1 -> W[1*2+1] = W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1
    W[1, 3] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1
    W[2, 4] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = 1
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
