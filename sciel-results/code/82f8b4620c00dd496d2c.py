import numpy as np

def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm.
    """
    # U: (4, 7) - Input A factors
    # V: (4, 7) - Input B factors
    # W: (4, 7) - Output C factors
    # Indices:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # r=0: M1 = (A00 + A11) * (B00 + B11)
    # A part: A00(idx 0) + A11(idx 3)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    # B part: B00(idx 0) + B11(idx 3)
    V[0, 0] = 1.0; V[3, 0] = 1.0
    # C part: C00(idx 0) + C11(idx 3)
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # r=1: M2 = (A10 + A11) * B00
    # A part: A10(idx 2) + A11(idx 3)
    U[2, 1] = 1.0; U[3, 1] = 1.0
    # B part: B00(idx 0)
    V[0, 1] = 1.0
    # C part: C10(idx 2) - C11(idx 3)
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # r=2: M3 = A00 * (B01 - B11)
    # A part: A00(idx 0)
    U[0, 2] = 1.0
    # B part: B01(idx 1) - B11(idx 3)
    V[1, 2] = 1.0; V[3, 2] = -1.0
    # C part: C01(idx 1) + C11(idx 3)
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # r=3: M4 = A11 * (B10 - B00)
    # A part: A11(idx 3)
    U[3, 3] = 1.0
    # B part: B10(idx 2) - B00(idx 0)
    V[2, 3] = 1.0; V[0, 3] = -1.0
    # C part: C00(idx 0) + C10(idx 2)
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # r=4: M5 = (A00 + A01) * B11
    # A part: A00(idx 0) + A01(idx 1)
    U[0, 4] = 1.0; U[1, 4] = 1.0
    # B part: B11(idx 3)
    V[3, 4] = 1.0
    # C part: -C00(idx 0) + C01(idx 1)
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # r=5: M6 = (A10 - A00) * (B00 + B01)
    # A part: A10(idx 2) - A00(idx 0)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    # B part: B00(idx 0) + B01(idx 1)
    V[0, 5] = 1.0; V[1, 5] = 1.0
    # C part: C11(idx 3)
    W[3, 5] = 1.0
    
    # r=6: M7 = (A01 - A11) * (B10 + B11)
    # A part: A01(idx 1) - A11(idx 3)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    # B part: B10(idx 2) + B11(idx 3)
    V[2, 6] = 1.0; V[3, 6] = 1.0
    # C part: C00(idx 0)
    W[0, 6] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
