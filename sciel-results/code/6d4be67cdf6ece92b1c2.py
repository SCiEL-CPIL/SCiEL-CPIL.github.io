import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks of B.
    This achieves rank 8 * 7 = 56.
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Strassen coefficients for 2x2x2
    # U: [A00, A01, A10, A11]
    # V: [B00, B01, B10, B11]
    # W: [C00, C01, C10, C11]
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # M1: C00+1, C11+1
        [0, 0, 1, -1], # M2: C10+1, C11-1
        [0, 1, 0, 1],  # M3: C01+1, C11+1
        [1, 0, 1, 0],  # M4: C00+1, C10+1
        [-1, 1, 0, 0], # M5: C00-1, C01+1
        [0, 0, 0, 1],  # M6: C11+1
        [1, 0, 0, 0]   # M7: C00+1
    ]
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Map local indices to global indices
    # A: i in [0,1], j in [0,1] -> idx = i*m + j
    # B: j in [0,1], k in [0,15] -> idx = j*p + k
    #    Block l (0..7) covers k in [2l, 2l+1]
    #    Local B00 -> k=2l, B01 -> k=2l+1, B10 -> k=16+2l, B11 -> k=16+2l+1
    # C: i in [0,1], k in [0,15] -> idx = k*n + i
    #    Block l covers k in [2l, 2l+1]
    #    Local C00 -> k=2l, C01 -> k=2l+1, C10 -> k=2l, C11 -> k=2l+1
    #    Wait, C is 2x16. C[i,k].
    #    W index is k*n + i.
    #    For block l, the output C[:, 2l:2l+2] is a 2x2 matrix.
    #    C00 corresponds to C[0, 2l] -> idx = (2l)*2 + 0 = 4l
    #    C01 corresponds to C[0, 2l+1] -> idx = (2l+1)*2 + 0 = 4l+2
    #    C10 corresponds to C[1, 2l] -> idx = (2l)*2 + 1 = 4l+1
    #    C11 corresponds to C[1, 2l+1] -> idx = (2l+1)*2 + 1 = 4l+3
    
    for l in range(8):
        for s in range(7):
            col = 7 * l + s
            
            # U: same for all blocks
            for a_idx in range(4):
                U[a_idx, col] = strassen_U[s][a_idx]
            
            # V: local B indices mapped to global
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = strassen_V[s][b_idx]
            
            # W: local C indices mapped to global
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = strassen_W[s][c_idx]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
