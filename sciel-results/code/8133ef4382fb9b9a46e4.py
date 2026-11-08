import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-55 exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 blocks and schoolbook for the 2x1x10 block.
    """
    n, m, p = DIMENSIONS
    # Total rank: 5 * 7 (Strassen) + 20 (Schoolbook 2x1x10) = 35 + 20 = 55
    rank = 55
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    # 5 blocks of 2x2x2 Strassen
    # Indices: i in {0,1}, j in {0,1}, k in {2b, 2b+1}
    for b in range(5):
        k_offset = 2 * b
        # Local indices for 2x2x2
        # U index: i * 2 + j (local) -> global i * 3 + j
        # V index: j * 2 + k (local) -> global j * 10 + (k_offset + k)
        # W index: k * 2 + i (local) -> global (k_offset + k) * 2 + i
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[0, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[3, col] = 1
        W[0, col] = 1; W[3, col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10
        U[2, col] = 1; U[3, col] = 1
        V[0, col] = 1
        W[2, col] = 1
        col += 1
        
        # M3: A00(B01-B11) -> C01
        U[0, col] = 1
        V[1, col] = 1; V[3, col] = -1
        W[1, col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C10, C00
        U[3, col] = 1
        V[2, col] = 1; V[0, col] = -1
        W[2, col] = 1; W[0, col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C01, C11
        U[0, col] = 1; U[1, col] = 1
        V[3, col] = 1
        W[1, col] = 1; W[3, col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[2, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[3, col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[1, col] = 1; U[3, col] = -1
        V[2, col] = 1; V[3, col] = 1
        W[0, col] = 1
        col += 1

    # 1 block of 2x1x10 Schoolbook
    # Indices: i in {0,1}, j=2, k in {0..9}
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, col] = 1
            V[2 * 10 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
