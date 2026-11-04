"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4
    matrix by a 4 by 6 matrix.
    
    Strategy:
    Split the output into three column blocks:
    1. C[:, 0:2] = A * B[:, 0:2] (2x4 * 4x2). Rank 14 (2x Strassen 7).
    2. C[:, 2:4] = A * B[:, 2:4] (2x4 * 4x2). Rank 14 (2x Strassen 7).
    3. C[:, 4:6] = A * B[:, 4:6] (2x4 * 4x2). Rank 8 (2x Schoolbook 4).
    Total Rank = 14 + 14 + 8 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 36
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, 36)
    V = [[0] * rank for _ in range(m * p)]  # (24, 36)
    W = [[0] * rank for _ in range(p * n)]  # (12, 36)
    
    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # U_coeffs[s]: dict {(i_loc, j_loc): coeff} for first matrix (2x2)
    # V_coeffs[s]: dict {(j_loc, k_loc): coeff} for second matrix (2x2)
    # W_coeffs[s]: dict {(i_loc, k_loc): coeff} for output (2x2)
    strassen_U = [
        {(0, 0): 1, (1, 1): 1},       # M1
        {(1, 0): 1, (1, 1): 1},       # M2
        {(0, 0): 1},                   # M3
        {(1, 1): 1},                   # M4
        {(0, 0): 1, (0, 1): 1},       # M5
        {(1, 0): 1, (0, 0): -1},      # M6
        {(0, 1): 1, (1, 1): -1},      # M7
    ]
    strassen_V = [
        {(0, 0): 1, (1, 1): 1},       # (Q00+Q11)
        {(0, 0): 1},                   # Q00
        {(0, 1): 1, (1, 1): -1},      # (Q01-Q11)
        {(1, 0): 1, (0, 0): -1},      # (Q10-Q00)
        {(1, 1): 1},                   # Q11
        {(0, 0): 1, (0, 1): 1},       # (Q00+Q01)
        {(1, 0): 1, (1, 1): 1},       # (Q10+Q11)
    ]
    strassen_W = [
        {(0, 0): 1, (1, 1): 1},       # M1 -> C00(+1), C11(+1)
        {(1, 0): 1, (1, 1): -1},      # M2 -> C10(+1), C11(-1)
        {(0, 1): 1, (1, 1): 1},       # M3 -> C01(+1), C11(+1)
        {(0, 0): 1, (1, 0): 1},       # M4 -> C00(+1), C10(+1)
        {(0, 0): -1, (0, 1): 1},      # M5 -> C00(-1), C01(+1)
        {(1, 1): 1},                   # M6 -> C11(+1)
        {(0, 0): 1},                   # M7 -> C00(+1)
    ]
    
    # Schoolbook coefficients for 2x2 * 2x2 -> 2x2
    schoolbook_U = [
        {(0, 0): 1},
        {(0, 1): 1},
        {(1, 0): 1},
        {(1, 1): 1},
    ]
    schoolbook_V = [
        {(0, 0): 1},
        {(1, 0): 1},
        {(0, 1): 1},
        {(1, 1): 1},
    ]
    schoolbook_W = [
        {(0, 0): 1},
        {(0, 1): 1},
        {(1, 0): 1},
        {(1, 1): 1},
    ]

    col = 0
    
    # Helper to add a block decomposition
    def add_block(a_block_idx, b_col_block_idx, use_strassen):
        nonlocal col
        # a_block_idx: 0 for A cols {0,1}, 1 for A cols {2,3}
        # b_col_block_idx: 0 for B cols {0,1}, 1 for B cols {2,3}, 2 for B cols {4,5}
        
        if use_strassen:
            U_coeffs = strassen_U
            V_coeffs = strassen_V
            W_coeffs = strassen_W
            num_terms = 7
        else:
            U_coeffs = schoolbook_U
            V_coeffs = schoolbook_V
            W_coeffs = schoolbook_W
            num_terms = 4
            
        # We are computing C_sub = A_sub * B_sub
        # A_sub is 2x2 (rows 0-1, cols a_block_idx*2 .. a_block_idx*2+1)
        # B_sub is 2x2 (rows 0-1 or 2-3? No, B is 4x2. We split B rows into 2x2 blocks)
        # Wait, the decomposition A(2x4) * B(4x2) = A1(2x2)B1(2x2) + A2(2x2)B2(2x2)
        # So we need to add TWO 2x2x2 decompositions for each 2x4x2 block.
        
        # Term 1: A1 * B1
        # A1: rows 0-1, cols a_block_idx*2 .. a_block_idx*2+1
        # B1: rows 0-1, cols b_col_block_idx*2 .. b_col_block_idx*2+1
        for s in range(num_terms):
            # Fill U for A1
            for (i_loc, j_loc), coeff in U_coeffs[s].items():
                i = i_loc
                j = a_block_idx * 2 + j_loc
                u_idx = i * m + j
                U[u_idx][col] += coeff
                
            # Fill V for B1
            for (j_loc, k_loc), coeff in V_coeffs[s].items():
                j = j_loc # B row local 0-1 corresponds to global 0-1
                k = b_col_block_idx * 2 + k_loc
                v_idx = j * p + k
                V[v_idx][col] += coeff
                
            # Fill W for C1
            for (i_loc, k_loc), coeff in W_coeffs[s].items():
                i = i_loc
                k = b_col_block_idx * 2 + k_loc
                w_idx = k * n + i
                W[w_idx][col] += coeff
                
            col += 1
            
        # Term 2: A2 * B2
        # A2: rows 0-1, cols (a_block_idx+1)*2 .. (a_block_idx+1)*2+1
        # B2: rows 2-3, cols b_col_block_idx*2 .. b_col_block_idx*2+1
        for s in range(num_terms):
            # Fill U for A2
            for (i_loc, j_loc), coeff in U_coeffs[s].items():
                i = i_loc
                j = (a_block_idx + 1) * 2 + j_loc
                u_idx = i * m + j
                U[u_idx][col] += coeff
                
            # Fill V for B2
            for (j_loc, k_loc), coeff in V_coeffs[s].items():
                j = 2 + j_loc # B row local 0-1 corresponds to global 2-3
                k = b_col_block_idx * 2 + k_loc
                v_idx = j * p + k
                V[v_idx][col] += coeff
                
            # Fill W for C2
            for (i_loc, k_loc), coeff in W_coeffs[s].items():
                i = i_loc
                k = b_col_block_idx * 2 + k_loc
                w_idx = k * n + i
                W[w_idx][col] += coeff
                
            col += 1

    # Block 1: C[:, 0:2] (b_col_block 0)
    # A is 2x4. We need to multiply by B[:, 0:2].
    # A = [A1, A2]. B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # We can choose a_block_idx? No, A is fixed.
    # The decomposition A(2x4)*B(4x2) uses A1 (cols 0-1) and A2 (cols 2-3).
    # So we call add_block for a_block 0 (A1) and a_block 1 (A2)?
    # No, add_block as defined above handles ONE a_block and splits B into B1, B2.
    # But we need A1*B1 AND A2*B2.
    # My add_block function currently does:
    #   1. A(a_block) * B_top
    #   2. A(a_block+1) * B_bottom
    # This is exactly what we need for a single 2x4x2 multiplication if we pass a_block=0.
    # Wait.
    # If I call add_block(0, 0, True):
    #   It computes A1 * B1 (where A1 is cols 0-1, B1 is rows 0-1)
    #   And A2 * B2 (where A2 is cols 2-3, B2 is rows 2-3)
    # This covers the entire A(2x4) * B(4x2) product!
    # So one call to add_block(0, b_col, use_strassen) handles the whole 2x4x2 block.
    
    # Let's re-verify the logic in add_block.
    # "Term 1: A1 * B1" uses a_block_idx for A cols.
    # "Term 2: A2 * B2" uses a_block_idx+1 for A cols.
    # Yes, this covers all 4 columns of A.
    
    # So:
    # Block 1: C[:, 0:2] -> b_col_block 0, use_strassen=True
    add_block(0, 0, True)
    
    # Block 2: C[:, 2:4] -> b_col_block 1, use_strassen=True
    add_block(0, 1, True)
    
    # Block 3: C[:, 4:6] -> b_col_block 2, use_strassen=False
    add_block(0, 2, False)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
