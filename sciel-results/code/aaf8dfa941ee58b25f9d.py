import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x2x14 multiplication into 7 independent 2x2x2 Strassen blocks.
    # Each block has rank 7. Total rank = 7 * 7 = 49.
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2
    # A forms (4 entries: a, b, c, d)
    A_forms = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [-1, 0, 1, 0],# M6: c-a
        [0, 1, 0, -1] # M7: b-d
    ]
    
    # B forms (4 entries: e, f, g, h)
    B_forms = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1],# M3: f-h
        [-1, 0, 1, 0],# M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # C forms (4 entries: p00, p10, p01, p11)
    C_forms = [
        [1, 0, 0, 1], # M1: p00 + p11
        [0, 1, 0, -1],# M2: p10 - p11
        [0, 0, 1, 1], # M3: p01 + p11
        [1, 1, 0, 0], # M4: p00 + p10
        [-1, 0, 1, 0],# M5: -p00 + p01
        [0, 0, 0, 1], # M6: p11
        [1, 0, 0, 0]  # M7: p00
    ]

    col = 0
    # 7 blocks of size 2x2 in the p dimension (14 = 7 * 2)
    for block_idx in range(7):
        # Local indices in B (m x p): 
        # B is 2 x 14. 
        # Block k uses columns 2k, 2k+1.
        # Rows are 0, 1.
        # So B indices are:
        # e: (0, 2k), f: (0, 2k+1)
        # g: (1, 2k), h: (1, 2k+1)
        b_idx_e = 0 * 14 + 2 * block_idx
        b_idx_f = 0 * 14 + 2 * block_idx + 1
        b_idx_g = 1 * 14 + 2 * block_idx
        b_idx_h = 1 * 14 + 2 * block_idx + 1
        b_indices = [b_idx_e, b_idx_f, b_idx_g, b_idx_h]
        
        # Local indices in C (p x n):
        # C is 14 x 2.
        # Block k uses rows 2k, 2k+1.
        # Columns are 0, 1.
        # So C indices are:
        # p00: (2k, 0), p01: (2k, 1)
        # p10: (2k+1, 0), p11: (2k+1, 1)
        c_idx_p00 = 2 * block_idx * 2 + 0
        c_idx_p01 = 2 * block_idx * 2 + 1
        c_idx_p10 = (2 * block_idx + 1) * 2 + 0
        c_idx_p11 = (2 * block_idx + 1) * 2 + 1
        c_indices = [c_idx_p00, c_idx_p10, c_idx_p01, c_idx_p11] # Order: p00, p10, p01, p11 to match C_forms

        for s in range(7):
            # U (n*m = 4)
            # A indices: a(0,0), b(0,1), c(1,0), d(1,1)
            u_indices = [0, 1, 2, 3]
            for i in range(4):
                U[u_indices[i], col] = A_forms[s][i]
                
            # V (m*p = 28)
            for i in range(4):
                val = B_forms[s][i]
                if val != 0:
                    V[b_indices[i], col] = val
                    
            # W (p*n = 28)
            for i in range(4):
                val = C_forms[s][i]
                if val != 0:
                    W[c_indices[i], col] = val
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
