"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # We use a decomposition based on Strassen's algorithm for 2x2x2 blocks.
    # Split m=4 into two blocks of 2.
    # For each block, split p=8 into four blocks of 2.
    # Each 2x2x2 block uses 7 rank-1 terms (Strassen).
    # Total terms = 2 * 4 * 7 = 56.
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=float)
    V = np.zeros((B_dim, rank), dtype=float)
    W = np.zeros((C_dim, rank), dtype=float)
    
    col = 0
    
    # Strassen formulas for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local A index (0,1,2,3) to coeff
    # v_coeffs: dict mapping local B index (0,1,2,3) to coeff
    # w_coeffs: dict mapping local C index (0,1,2,3) to coeff
    
    # Local A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (row major: 00, 01, 10, 11)
    # Local C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (row major: 00, 01, 10, 11)
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # C10 gets M2, C11 gets -M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # C01 gets M3, C11 gets M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C00 gets M4, C10 gets M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C00 gets -M5, C01 gets M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C11 gets M6
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1}) # C00 gets M7, C11 gets -M7
    ]
    
    # Iterate over the 2 main blocks (m-split)
    for m_block in range(2):
        # m_block 0 uses A cols 0,1 and B rows 0,1
        # m_block 1 uses A cols 2,3 and B rows 2,3
        
        # Iterate over the 4 sub-blocks (p-split)
        for p_block in range(4):
            # p_block s uses B cols 2s, 2s+1 and C rows 2s, 2s+1
            
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
                # Map local indices to global indices
                
                # U (A)
                # Local A index l (0..3) corresponds to i = l//2, j_loc = l%2
                # Global j = m_block * 2 + j_loc
                # Global a = i * m + j
                for l, coeff_u in u_loc.items():
                    i = l // 2
                    j_loc = l % 2
                    j = m_block * 2 + j_loc
                    a = i * m + j
                    U[a, col] += coeff_u
                
                # V (B)
                # Local B index l (0..3) corresponds to j_loc = l//2, k_loc = l%2
                # Global j = m_block * 2 + j_loc
                # Global k = p_block * 2 + k_loc
                # Global b = j * p + k
                for l, coeff_v in v_loc.items():
                    j_loc = l // 2
                    k_loc = l % 2
                    j = m_block * 2 + j_loc
                    k = p_block * 2 + k_loc
                    b = j * p + k
                    V[b, col] += coeff_v
                
                # W (C)
                # Local C index l (0..3) corresponds to k_loc = l//2, i = l%2
                # Global k = p_block * 2 + k_loc
                # Global i = i
                # Global c = k * n + i
                for l, coeff_w in w_loc.items():
                    k_loc = l // 2
                    i = l % 2
                    k = p_block * 2 + k_loc
                    c = k * n + i
                    W[c, col] += coeff_w
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
