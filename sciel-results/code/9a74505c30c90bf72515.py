"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen(X_idx, Y_idx, Z_idx, offset):
        # X_idx: [(r,c), (r,c), (r,c), (r,c)] for X00, X01, X10, X11
        # Y_idx: [(r,c), (r,c), (r,c), (r,c)] for Y00, Y01, Y10, Y11
        # Z_idx: [(r,c), (r,c), (r,c), (r,c)] for Z00, Z01, Z10, Z11
        
        terms = [
            # x_coeffs, y_coeffs, z_coeffs
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # M3
            ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
        ]
        
        for t in range(7):
            col = offset + t
            xc, yc, zc = terms[t]
            
            for l in range(4):
                if xc[l] != 0:
                    r, c = X_idx[l]
                    U[r * m + c, col] += xc[l]
                if yc[l] != 0:
                    r, c = Y_idx[l]
                    V[r * p + c, col] += yc[l]
                if zc[l] != 0:
                    r, c = Z_idx[l]
                    W[c * n + r, col] += zc[l]

    # Define indices for blocks
    # A is 2x4. A_L is cols 0,1. A_R is cols 2,3.
    AL = [(0,0), (0,1), (1,0), (1,1)]
    AR = [(0,2), (0,3), (1,2), (1,3)]
    
    # B is 4x4.
    # B_TL: rows 0,1 cols 0,1
    BTL = [(0,0), (0,1), (1,0), (1,1)]
    # B_TR: rows 0,1 cols 2,3
    BTR = [(0,2), (0,3), (1,2), (1,3)]
    # B_BL: rows 2,3 cols 0,1
    BBL = [(2,0), (2,1), (3,0), (3,1)]
    # B_BR: rows 2,3 cols 2,3
    BBR = [(2,2), (2,3), (3,2), (3,3)]
    
    # C is 2x4.
    # C_L: cols 0,1
    CL = [(0,0), (0,1), (1,0), (1,1)]
    # C_R: cols 2,3
    CR = [(0,2), (0,3), (1,2), (1,3)]
    
    # 1. AL * BTL -> CL
    add_strassen(AL, BTL, CL, 0)
    # 2. AR * BBL -> CL
    add_strassen(AR, BBL, CL, 7)
    # 3. AL * BTR -> CR
    add_strassen(AL, BTR, CR, 14)
    # 4. AR * BBR -> CR
    add_strassen(AR, BBR, CR, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
