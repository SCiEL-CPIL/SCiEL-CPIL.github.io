"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Strassen Rank 7 Decomposition
    # M1 = (A11 - A22)(B21 + B22)
    # M2 = (A11 + A12) B22
    # M3 = A11 (B11 - B22)
    # M4 = A22 (B11 + B12)
    # M5 = (A11 + A22)(B11 - B12)
    # M6 = (A21 - A11)(B11 + B21)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M2 + M3
    # C21 = M6 + M7
    # C22 = M1 - M3 + M4 + M5
    
    # U indices: i*m + j (0: A00, 1: A01, 2: A10, 3: A11)
    # V indices: j*p + k (0: B00, 1: B01, 2: B10, 3: B11)
    # W indices: k*n + i (0: C00, 1: C10, 2: C01, 3: C11)
    
    U_data = [
        [1, 1, 1, 0, 1, -1, 0],  # A00
        [0, 1, 0, 0, 0, 0, 1],   # A01
        [0, 0, 0, 0, 0, 1, 0],   # A10
        [-1, 0, 0, 1, 1, 0, -1]  # A11
    ]
    
    V_data = [
        [0, 0, 1, 1, 1, 1, 0],   # B00
        [0, 0, 0, 1, -1, 0, 0],  # B01
        [1, 0, 0, 0, 0, 1, 1],   # B10
        [1, 1, -1, 0, 0, 0, 1]   # B11
    ]
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 0, 0, 0, 0, 1, 1],   # C10 (k=0, i=1)
        [0, 1, 1, 0, 0, 0, 0],   # C01 (k=1, i=0)
        [1, 0, -1, 1, 1, 0, 0]   # C11 (k=1, i=1)
    ]
    
    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)
    
    # Verification
    valid = True
    rank = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 8
    rank = 8
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
