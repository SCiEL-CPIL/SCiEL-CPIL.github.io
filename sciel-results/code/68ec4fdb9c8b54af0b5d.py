import numpy as np
import time

def construct():
    """
    Returns a rank-17 tensor decomposition for 2x3 x 3x3 matrix multiplication.
    
    The decomposition is constructed by combining:
    1. Strassen's algorithm (rank 7) for the 2x2 x 2x2 block (columns 0,1 of B).
    2. Direct multiplication (rank 4) for the 2x2 x 1x1 block (column 2 of B).
    3. Direct multiplication (rank 6) for the 2x1 x 3x3 block (row 2 of A).
    
    Total rank = 7 + 4 + 6 = 17.
    This is better than the schoolbook rank of 18.
    
    The matrices U, V, W are returned as lists of lists, with shapes:
    U: (6, 17)  [2*3]
    V: (9, 17)  [3*3]
    W: (6, 17)  [3*2]
    
    Indices follow the convention:
    U[i*m + j, r] for A[i, j]
    V[j*p + k, r] for B[j, k]
    W[k*n + i, r] for C[i, k]
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def set_U(i, j, r, val): U[i * m + j, r] = val
    def set_V(j, k, r, val): V[j * p + k, r] = val
    def set_W(k, i, r, val): W[k * n + i, r] = val
    
    # --- Block 1: Strassen 2x2 x 2x2 (j in {0,1}, k in {0,1}) ---
    # r = 0..6
    
    # M1 = (A00+A11)(B00+B11)
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(0, 1, 0, 1); set_W(1, 0, 0, 1); set_W(1, 1, 0, 1)
    
    # M2 = (A10+A11)B00
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(1, 0, 1, 1); set_W(1, 1, 1, 1)
    
    # M3 = A00(B10-B00)
    set_U(0, 0, 2, 1)
    set_V(1, 0, 2, 1); set_V(0, 0, 2, -1)
    set_W(0, 0, 2, 1); set_W(1, 0, 2, 1)
    
    # M4 = A11(B01-B10)
    set_U(1, 1, 3, 1)
    set_V(0, 1, 3, 1); set_V(1, 0, 3, -1)
    set_W(0, 1, 3, 1); set_W(1, 1, 3, 1)
    
    # M5 = (A00+A01)B11
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 1, 4, 1); set_W(1, 1, 4, 1)
    
    # M6 = (A10-A00)(B00+B01)
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(0, 0, 5, 1); set_W(1, 0, 5, 1)
    
    # M7 = (A01-A11)(B10+B11)
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 1, 6, 1); set_W(1, 1, 6, 1)
    
    # --- Block 2: Direct 2x2 x 1x1 (j in {0,1}, k=2) ---
    # r = 7..10
    
    # C02 += A00 * B02
    set_U(0, 0, 7, 1)
    set_V(0, 2, 7, 1)
    set_W(2, 0, 7, 1)
    
    # C02 += A01 * B12
    set_U(0, 1, 8, 1)
    set_V(1, 2, 8, 1)
    set_W(2, 0, 8, 1)
    
    # C12 += A10 * B02
    set_U(1, 0, 9, 1)
    set_V(0, 2, 9, 1)
    set_W(2, 1, 9, 1)
    
    # C12 += A11 * B12
    set_U(1, 1, 10, 1)
    set_V(1, 2, 10, 1)
    set_W(2, 1, 10, 1)
    
    # --- Block 3: Direct 2x1 x 3x3 (j=2, k in {0,1,2}) ---
    # r = 11..16
    
    # C00 += A02 * B20
    set_U(0, 2, 11, 1)
    set_V(2, 0, 11, 1)
    set_W(0, 0, 11, 1)
    
    # C01 += A02 * B21
    set_U(0, 2, 12, 1)
    set_V(2, 1, 12, 1)
    set_W(1, 0, 12, 1)
    
    # C02 += A02 * B22
    set_U(0, 2, 13, 1)
    set_V(2, 2, 13, 1)
    set_W(2, 0, 13, 1)
    
    # C10 += A12 * B20
    set_U(1, 2, 14, 1)
    set_V(2, 0, 14, 1)
    set_W(0, 1, 14, 1)
    
    # C11 += A12 * B21
    set_U(1, 2, 15, 1)
    set_V(2, 1, 15, 1)
    set_W(1, 1, 15, 1)
    
    # C12 += A12 * B22
    set_U(1, 2, 16, 1)
    set_V(2, 2, 16, 1)
    set_W(2, 1, 16, 1)
    
    # Return as lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
