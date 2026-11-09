import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Rank 49 for 4x4x4 Strassen + 16 for schoolbook on 5th column = 65
    rank = 49 + 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- 2x2x2 Strassen definitions ---
    # For multiplying X (2x2) by Y (2x2) = Z (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Outer Strassen for 4x4x4:
    # Split A (4x4) into blocks A00,A01,A10,A11 (each 2x2)
    # Split B (4x4) into blocks B00,B01,B10,B11 (each 2x2)
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each P_k is a 2x2 matrix = product of two 2x2 linear combinations
    # We decompose each P_k using inner Strassen (7 scalar products)
    
    # Define outer Strassen block combinations
    # Each entry: (list of (block_row, block_col, coeff) for A, list of (block_row, block_col, coeff) for B)
    outer_products = [
        # P1: (A00+A11)(B00+B11)
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # P2: (A10+A11)B00
        ([((1,0,1), (1,1,1))], [((0,0,1))]),
        # P3: A00(B01-B11)
        ([((0,0,1))], [((0,1,1), (1,1,-1))]),
        # P4: A11(B10-B00)
        ([((1,1,1))], [((1,0,1), (0,0,-1))]),
        # P5: (A00+A01)B11
        ([((0,0,1), (0,1,1))], [((1,1,1))]),
        # P6: (A10-A00)(B00+B01)
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))]),
        # P7: (A01-A11)(B10+B11)
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))])
    ]
    
    # Inner Strassen: for 2x2 X * 2x2 Y
    # X entries: X00, X01, X10, X11
    # Y entries: Y00, Y01, Y10, Y11
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # For each inner term, define X linear form and Y linear form
    # X form: list of (x_row, x_col, coeff)
    # Y form: list of (y_row, y_col, coeff)
    inner_terms = [
        # (X_form, Y_form)
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),  # M1
        ([((1,0,1), (1,1,1))], [((0,0,1))]),             # M2
        ([((0,0,1))], [((0,1,1), (1,1,-1))]),            # M3
        ([((1,1,1))], [((1,0,1), (0,0,-1))]),            # M4
        ([((0,0,1), (0,1,1))], [((1,1,1))]),             # M5
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))]),   # M6
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))])    # M7
    ]
    
    # Output combination for inner Strassen:
    # Z00 = M1 + M4 - M5 + M7  -> terms [0, 3, -1*4, 6] with coeffs [1,1,-1,1]
    # Z01 = M3 + M5             -> terms [2, 4] with coeffs [1,1]
    # Z10 = M2 + M4             -> terms [1, 3] with coeffs [1,1]
    # Z11 = M1 - M2 + M3 + M6   -> terms [0, -1*1, 2, 5] with coeffs [1,-1,1,1]
    inner_output = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)]
    }
    
    # Output combination for outer Strassen:
    # C00_block = P1 + P4 - P5 + P7
    # C01_block = P3 + P5
    # C10_block = P2 + P4
    # C11_block = P1 - P2 + P3 + P6
    outer_output = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],  # P1, P4, -P5, P7
        (0,1): [(2,1), (4,1)],                   # P3, P5
        (1,0): [(1,1), (3,1)],                   # P2, P4
        (1,1): [(0,1), (1,-1), (2,1), (5,1)]     # P1, -P2, P3, P6
    }
    
    # Helper: get global A index for block (bi,bj) and local (li,lj)
    def a_global(bi, bj, li, lj):
        return (bi*2 + li) * m + (bj*2 + lj)
    
    # Helper: get global B index for block (bi,bj) and local (li,lj)
    # B is m x p = 4 x 5, but we only use columns 0..3 for the 4x4x4 part
    def b_global(bi, bj, li, lj):
        return (bi*2 + li) * p + (bj*2 + lj)
    
    # Helper: get global W index for C block (ci,cj) and local (li,lj)
    # C is n x p = 4 x 5, we use columns 0..3
    def w_global(ci, cj, li, lj):
        k = cj*2 + lj  # column in C (0..3)
        i = ci*2 + li  # row in C (0..3)
        return k * n + i
    
    # Build the 49 columns for 4x4x4 Strassen
    col = 0
    for p_idx in range(7):
        a_blocks, b_blocks = outer_products[p_idx]
        # a_blocks is a list with one element: list of (bi, bj, coeff)
        a_combo = a_blocks[0]
        b_combo = b_blocks[0]
        
        for m_idx in range(7):
            x_form, y_form = inner_terms[m_idx]
            
            # Build U contribution for this column
            # U[i*m+j, col] = sum over A blocks and inner X positions
            # For each (x_row, x_col, x_coeff) in x_form:
            #   For each (bi, bj, a_coeff) in a_combo:
            #     global_i = bi*2 + x_row, global_j = bj*2 + x_col
            #     U[global_i * m + global_j, col] += a_coeff * x_coeff
            
            u_vals = {}
            for (x_r, x_c, x_coeff) in x_form:
                for (bi, bj, a_coeff) in a_combo:
                    gi = bi * 2 + x_r
                    gj = bj * 2 + x_c
                    idx = gi * m + gj
                    u_vals[idx] = u_vals.get(idx, 0) + a_coeff * x_coeff
            
            for idx, val in u_vals.items():
                if val != 0:
                    U[idx, col] += val
            
            # Build V contribution
            # V[j*p+k, col] = sum over B blocks and inner Y positions
            # For each (y_row, y_col, y_coeff) in y_form:
            #   For each (bi, bj, b_coeff) in b_combo:
            #     global_j = bi*2 + y_row, global_k = bj*2 + y_col
            #     V[global_j * p + global_k, col] += b_coeff * y_coeff
            
            v_vals = {}
            for (y_r, y_c, y_coeff) in y_form:
                for (bi, bj, b_coeff) in b_combo:
                    gj = bi * 2 + y_r
                    gk = bj * 2 + y_c
                    idx = gj * p + gk
                    v_vals[idx] = v_vals.get(idx, 0) + b_coeff * y_coeff
            
            for idx, val in v_vals.items():
                if val != 0:
                    V[idx, col] += val
            
            # Build W contribution
            # This is trickier. The inner Strassen produces a 2x2 result Z.
            # Z[zi_r, zi_c] gets contributions from inner terms.
            # Then P_k[zi_r, zi_c] is part of the outer result.
            # The outer result C_block[ci, cj] gets P_k with some sign.
            # C[ci*2+zi_r, cj*2+zi_c] = sum over outer contributions
            
            # For this specific (p_idx, m_idx), determine which C entries it affects
            # m_idx affects Z entries according to inner_output
            # p_idx affects C blocks according to outer_output
            
            for (zi_r, zi_c), m_contribs in inner_output.items():
                for (mi, m_coeff) in m_contribs:
                    if mi != m_idx:
                        continue
                    # This inner term m_idx contributes to Z[zi_r, zi_c] with coeff m_coeff
                    # Now, P_{p_idx}[zi_r, zi_c] contributes to C blocks
                    for (ci, cj), p_contribs in outer_output.items():
                        for (pi, p_coeff) in p_contribs:
                            if pi != p_idx:
                                continue
                            # C[ci*2+zi_r, cj*2+zi_c] gets contribution
                            i_global = ci * 2 + zi_r
                            k_global = cj * 2 + zi_c
                            w_idx = k_global * n + i_global
                            total_coeff = m_coeff * p_coeff
                            if total_coeff != 0:
                                W[w_idx, col] += total_coeff
            
            col += 1
    
    # --- Schoolbook for k=4 (5th column of B) ---
    # For each i in 0..3, j in 0..3: one column
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
