import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    The 2x12 matrix B is split into six 2x2 column-pairs. Each pair is
    multiplied by the 2x2 matrix A using Strassen's 7-product algorithm.
    Total rank = 6 * 7 = 42.
    
    Strassen formulas for A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[C00,C01],[C10,C11]]:
      M1 = (a+d)(e+h)
      M2 = (c+d)e
      M3 = a(f-h)
      M4 = d(g-e)
      M5 = (a+b)h
      M6 = (c-a)(e+f)
      M7 = (b-d)(g+h)
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for t in range(num_blocks):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A00+A11)*(B00+B11)
        c = base + 0
        U[0, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0
        
        # M2 = (A10+A11)*B00
        c = base + 1
        U[2, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        W[k0 * n + 1, c] = 1.0
        W[k1 * n + 1, c] = -1.0
        
        # M3 = A00*(B01-B11)
        c = base + 2
        U[0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        V[1 * p + k1, c] = -1.0
        W[k1 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0
        
        # M4 = A11*(B10-B00)
        c = base + 3
        U[3, c] = 1.0
        V[1 * p + k0, c] = 1.0
        V[0 * p + k0, c] = -1.0
        W[k0 * n + 0, c] = 1.0
        W[k0 * n + 1, c] = 1.0
        
        # M5 = (A00+A01)*B11
        c = base + 4
        U[0, c] = 1.0
        U[1, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = -1.0
        W[k1 * n + 0, c] = 1.0
        
        # M6 = (A10-A00)*(B00+B01)
        c = base + 5
        U[2, c] = 1.0
        U[0, c] = -1.0
        V[0 * p + k0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        W[k1 * n + 1, c] = 1.0
        
        # M7 = (A01-A11)*(B10+B11)
        c = base + 6
        U[1, c] = 1.0
        U[3, c] = -1.0
        V[1 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
