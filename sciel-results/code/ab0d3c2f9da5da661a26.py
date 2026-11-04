import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0

def _time_left():
    return max(0.0, _DEADLINE - time.time())

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: (4, 7)
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    
    return U, V, W

def _build_4x4x8():
    """
    Constructs the rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    This is formed by two independent 4x4x4 Strassen-Kronecker blocks.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Build 4x4x4 decomposition (Rank 49)
    # We can use the loop method or Kronecker + Permutation.
    # Loop method is simple and fast enough for 16x49.
    U4 = np.zeros((16, 49), dtype=np.int16)
    V4 = np.zeros((16, 49), dtype=np.int16)
    W4 = np.zeros((16, 49), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i * 4 + j, c] = U2[(i // 2) * 2 + (j // 2), c0] * U2[(i % 2) * 2 + (j % 2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j * 4 + k, c] = V2[(j // 2) * 2 + (k // 2), c0] * V2[(j % 2) * 2 + (k % 2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k * 4 + i, c] = W2[(k // 2) * 2 + (i // 2), c0] * W2[(k % 2) * 2 + (i % 2), c1]
                    
    # Assemble 4x4x8 decomposition (Rank 98)
    # U is 16x98
    # V is 32x98 (4 rows for A part? No, V is for B. B is 4x8. Flattened 32x1)
    # W is 32x98 (C is 4x8. Flattened 32x1. Index k*n+i. k is 0..3, i is 0..3)
    
    U = np.zeros((16, 98), dtype=np.int16)
    V = np.zeros((32, 98), dtype=np.int16)
    W = np.zeros((32, 98), dtype=np.int16)
    
    # Block 0: Columns 0-3 of output
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = k_local # 0..3
            V[j * 8 + k_global, 0:49] = V4[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(4):
            W[k_local * 4 + i, 0:49] = W4[k_local * 4 + i, :]
            
    # Block 1: Columns 4-7 of output
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local # 4..7
            V[j * 8 + k_global, 49:98] = V4[j * 4 + k_local, :]
    for k_local in range(4):
        for i in range(4):
            W[(4 + k_local) * 4 + i, 49:98] = W4[k_local * 4 + i, :]
            
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x8.
    Returns dict with 'U', 'V', 'W' matrices.
    """
    U, V, W = _build_4x4x8()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
