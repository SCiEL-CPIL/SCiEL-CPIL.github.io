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
    
    # Precompute Strassen matrices as numpy arrays for faster broadcasting
    S_U = np.array(strassen_U).T  # Shape (4, 7)
    S_V = np.array(strassen_V).T  # Shape (4, 7)
    S_W = np.array(strassen_W).T  # Shape (4, 7)
    
    for l in range(8):
        col_start = 7 * l
        col_end = 7 * l + 7
        
        # U: same for all blocks, just repeat the 7 columns
        U[:, col_start:col_end] = S_U
        
        # V: local B indices mapped to global
        # B rows: 2l, 2l+1, 16+2l, 16+2l+1
        # S_V rows correspond to B00, B01, B10, B11
        V[2*l:2*l+2, col_start:col_end] = S_V[:2, :]
        V[16+2*l:16+2*l+2, col_start:col_end] = S_V[2:, :]
        
        # W: local C indices mapped to global
        # C rows: 4l, 4l+2, 4l+1, 4l+3
        # S_W rows correspond to C00, C01, C10, C11
        # Note: The order in S_W is C00, C01, C10, C11
        # Global indices:
        # C00 -> 4l
        # C01 -> 4l+2
        # C10 -> 4l+1
        # C11 -> 4l+3
        
        W[4*l, col_start:col_end] = S_W[0, :]
        W[4*l+2, col_start:col_end] = S_W[1, :]
        W[4*l+1, col_start:col_end] = S_W[2, :]
        W[4*l+3, col_start:col_end] = S_W[3, :]
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Convert to lists as required
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
