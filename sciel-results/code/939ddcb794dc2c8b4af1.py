import numpy as np

def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm.
    
    Indexing:
    U rows: i*2 + j  (A[i,j])
    V rows: j*2 + k  (B[j,k])
    W rows: k*2 + i  (C[i,k])
    """
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # r=0: M0 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0  # C00 += 1
    W[3, 0] = 1.0  # C11 += 1
    
    # r=1: M1 = (A10 + A11) * B00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0  # C10 += 1
    W[3, 1] = -1.0 # C11 -= 1
    
    # r=2: M2 = A00 * (B01 - B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0  # C01 += 1
    W[3, 2] = 1.0  # C11 += 1
    
    # r=3: M3 = A11 * (B10 - B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0  # C00 += 1
    W[1, 3] = 1.0  # C10 += 1
    
    # r=4: M4 = (A00 + A01) * B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0 # C00 -= 1
    W[2, 4] = 1.0  # C01 += 1
    
    # r=5: M5 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[1, 5] = 1.0  # C10 += 1
    
    # r=6: M6 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = -1.0 # C00 -= 1
    W[3, 6] = 1.0  # C11 += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
