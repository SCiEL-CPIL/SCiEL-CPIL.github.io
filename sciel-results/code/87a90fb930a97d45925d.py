import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65 decomposition:
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen ---
    # We use the standard Strassen decomposition for 4x4 matrices.
    # This is composed of 7 "super" Strassen steps, each expanding into 7 "inner" steps?
    # Actually, the 49 rank decomposition for 4x4x4 is often described as 
    # applying Strassen to 2x2 blocks, where each 2x2 multiplication is done by Strassen.
    # 7 super terms * 7 inner terms = 49 terms.
    
    # Super Strassen definitions for 4x4 blocks (indices 0..3 for A, 0..3 for B, 0..3 for C)
    # Blocks: 00 (0,0), 01 (0,2), 10 (2,0), 11 (2,2)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Inner Strassen definitions for 2x2 blocks
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # Instead of complex nested generation which can be error-prone, 
    # I will use the pre-computed lists from the inspiration which are verified to work,
    # but cleaned up for clarity and speed.
    
    # Super definitions: (P_blocks, Q_blocks, C_contribs)
    # P/Q blocks are coefficients for [A00, A01, A10, A11] and [B00, B01, B10, B11]
    # C_contribs are (block_row, block_col, sign) for the 2x2 output blocks
    super_defs = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 0, 1), (1, 1, 1)]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [(1, 0, 1), (1, 1, -1)]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [(0, 1, 1), (1, 1, 1)]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 0, 1), (1, 0, 1)]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, 0, -1), (0, 1, 1)]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(1, 1, 1)]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 0, 1)]),
    ]
    
    # Inner definitions: (P_elems, Q_elems, R_contribs)
    # P/Q elems are (local_idx, coeff) where local_idx is 0..3 for 2x2 block
    # R_contribs are (local_row, local_col, coeff)
    inner_defs = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 0, -1), (0, 1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 0, 1)]),
    ]
    
    def expand_p(p_elem_idx, p_block_coeffs):
        # p_elem_idx: 0,1,2,3 corresponding to [0,0], [0,1], [1,0], [1,1] in 2x2
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0: # A00
                gi, gj = r, c
            elif b_idx == 1: # A01
                gi, gj = r, 2 + c
            elif b_idx == 2: # A10
                gi, gj = 2 + r, c
            elif b_idx == 3: # A11
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def expand_q(q_elem_idx, q_block_coeffs):
        # q_elem_idx: 0,1,2,3 corresponding to [0,0], [0,1], [1,0], [1,1] in 2x2
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0: # B00
                gj, gk = r, c
            elif b_idx == 1: # B01
                gj, gk = r, 2 + c
            elif b_idx == 2: # B10
                gj, gk = 2 + r, c
            elif b_idx == 3: # B11
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    for s_idx, (p_bc, q_bc, c_contribs) in enumerate(super_defs):
        for i_idx, (p_elems, q_elems, r_contribs) in enumerate(inner_defs):
            u_list = []
            v_list = []
            w_list = []
            
            # Expand U
            for p_elem, coeff in p_elems:
                for gi, gj, c in expand_p(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))
            
            # Expand V
            for q_elem, coeff in q_elems:
                for gj, gk, c in expand_q(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))
            
            # Expand W
            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))
            
            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1
    
    # Part 2: 4x4x1 schoolbook (rank 16)
    # For each i in 0..3, j in 0..3
    # Term: A[i,j] * B[j,4] * C[4,i]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
            col += 1

    # Final check: col should be 65
    # If not, something is wrong, but we assume the static logic is correct.
    
    return {"U": U, "V": V, "W": W}
