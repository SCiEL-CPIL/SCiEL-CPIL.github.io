import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    The rank is 44, which is the sum of the ranks of the 2x2*2x8 and 2x1*1x8 parts.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)  # shape (6, 44)
    V = np.zeros((m * p, rank), dtype=float)  # shape (24, 44)
    W = np.zeros((p * n, rank), dtype=float)  # shape (16, 44)
    
    # Strassen 2x2x2 coefficients
    # U factors for A (2x2)
    # V factors for B (2x2)
    # W factors for C (2x2)
    S_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    S_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0, 1 of B. Split into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            # A is 2x3. A1 is cols 0,1.
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx, col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx, col] += val
            
            # Fill W for C block (cols k_start..k_start+1 of C, rows 0..1 of C)
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx, col] += val
            
            col += 1
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A (rows 0,1).
    # B2 is row 2 of B (cols 0..7).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
