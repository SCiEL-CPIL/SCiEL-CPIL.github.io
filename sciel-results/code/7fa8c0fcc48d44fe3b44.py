import numpy as np

def get_strassen_factors():
    # U, V, W are 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    # M0: A00(B00+B11)
    U[0, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    # W?
    
    # M1: (A00+A10)B11
    U[0, 1] = 1
    U[2, 1] = 1
    V[3, 1] = 1
    
    # M2: (A01+A11)B00
    U[1, 2] = 1
    U[3, 2] = 1
    V[0, 2] = 1
    
    # M3: A11(B10+B00)
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = 1
    
    # M4: (A00+A11)(B00+B11)
    U[0, 4] = 1
    U[3, 4] = 1
    V[0, 4] = 1
    V[3, 4] = 1
    
    # M5: (A01-A11)(B10+B11)
    U[1, 5] = 1
    U[3, 5] = -1
    V[2, 5] = 1
    V[3, 5] = 1
    
    # M6: (A00-A10)(B00+B01)
    U[0, 6] = 1
    U[2, 6] = -1
    V[0, 6] = 1
    V[1, 6] = 1
    
    # Now determine W such that C = U * (A^T V) * W?
    # Tensor T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # We need to find W.
    # We can solve for W column by column?
    # Or use the known formulas for C.
    
    # Let's just use the known formulas and fill W.
    # C00 = M4 + M3 - M1 + M2
    W[0, 0] = 0
    W[0, 1] = -1
    W[0, 2] = 1
    W[0, 3] = 1
    W[0, 4] = 1
    W[0, 5] = 0
    W[0, 6] = 0
    
    # C01 = M0 + M6
    W[1, 0] = 1
    W[1, 1] = 0
    W[1, 2] = 0
    W[1, 3] = 0
    W[1, 4] = 0
    W[1, 5] = 0
    W[1, 6] = 1
    
    # C10 = M1 + M5
    W[2, 0] = 0
    W[2, 1] = 1
    W[2, 2] = 0
    W[2, 3] = 0
    W[2, 4] = 0
    W[2, 5] = 1
    W[2, 6] = 0
    
    # C11 = M4 + M2 - M0 + M5
    W[3, 0] = -1
    W[3, 1] = 0
    W[3, 2] = 1
    W[3, 3] = 0
    W[3, 4] = 1
    W[3, 5] = 1
    W[3, 6] = 0
    
    return U, V, W
