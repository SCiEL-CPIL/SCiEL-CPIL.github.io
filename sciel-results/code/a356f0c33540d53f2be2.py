import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Block-diagonal decomposition:
    # Part 1: 1x4x4 (Row 0 of A, Row 0 of C)
    # Part 2: 2x4x4 (Rows 1-2 of A, Rows 1-2 of C)
    
    # Rank of 1xn xn is n^2. Rank(1x4x4) = 16.
    # Rank of 2x4x4 is 2 * Rank(1x4x4) = 32.
    # Total Rank = 48.
    rank = 48
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 1x4x4 ---
    # A is 1x4 (indices 0-3)
    # B is 4x4 (indices 0-15)
    # C is 1x4 (indices 0, 3, 6, 9 in W)
    
    term_offset = 0
    for k in range(4):  # A col / B row
        for j in range(4):  # B col / C col
            t = term_offset + k * 4 + j
            a_idx = k
            b_idx = k * 4 + j
            c_idx = j * n  # Row 0 of C: j * 3 + 0
            
            U[a_idx, t] = 1
            V[b_idx, t] = 1
            W[c_idx, t] = 1
    
    # --- Part 2: 2x4x4 ---
    # A is 2x4 (rows 1,2 -> indices 4-7, 8-11)
    # B is 4x4 (indices 0-15)
    # C is 2x4 (rows 1,2 -> indices 3j+1, 3j+2 in W)
    
    term_offset = 16
    for r in range(1, 3):  # Row index in A/C (1 and 2)
        for k in range(4):  # A col / B row
            for j in range(4):  # B col / C col
                t = term_offset + (r - 1) * 16 + k * 4 + j
                a_idx = r * m + k
                b_idx = k * 4 + j
                c_idx = j * n + r
                
                U[a_idx, t] = 1
                V[b_idx, t] = 1
                W[c_idx, t] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
