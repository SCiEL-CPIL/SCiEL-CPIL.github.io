import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen coefficients
    # (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs is dict { (i,j): val }
    # B_coeffs is dict { (i,j): val }
    # C_coeffs is dict { (i,j): val }
    
    strassen_blocks = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # P1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}), # P2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}), # P3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # P4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # P5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # P6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}) # P7
    ]
    
    # We have 8 sub-problems
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x8. Split into B_T (rows 0,1) and B_B (rows 2,3).
    # C is 2x8.
    
    # Sub-problem (a_idx, b_idx) where a_idx in [0,1] (L/R), b_idx in [0,1] (T/B)
    # a_idx 0 -> A cols 0,1. a_idx 1 -> A cols 2,3.
    # b_idx 0 -> B rows 0,1. b_idx 1 -> B rows 2,3.
    # B is 2x8 in each sub-problem. Split into 4 blocks of 2x2.
    # Block q in [0,3] -> B cols 2q, 2q+1.
    
    for a_idx in range(2):
        for b_idx in range(2):
            for q in range(4):
                for A_c, B_c, C_c in strassen_blocks:
                    # Fill U
                    for (i, j), val in A_c.items():
                        # Local i, j in 0..1
                        # Global A index: row i, col (a_idx*2 + j)
                        global_a_idx = i * m + (a_idx * 2 + j)
                        U[global_a_idx, col] += val
                        
                    # Fill V
                    for (i, j), val in B_c.items():
                        # Local i, j in 0..1
                        # Global B index: row (b_idx*2 + i), col (q*2 + j)
                        global_b_idx = (b_idx * 2 + i) * p + (q * 2 + j)
                        V[global_b_idx, col] += val
                        
                    # Fill W
                    for (i, j), val in C_c.items():
                        # Local i, j in 0..1
                        # Global C index: row i, col (q*2 + j)
                        # W index is k*n + i, where k is col, i is row
                        global_c_idx = (q * 2 + j) * n + i
                        W[global_c_idx, col] += val
                        
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
