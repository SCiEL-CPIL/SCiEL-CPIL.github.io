"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 145.0

    # Rank 35: 19 (Laderman for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Helper to add a rank-1 term
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
    
    # --- Part 1: 4x4 * 4x4 using Laderman (Rank 19) ---
    
    # Laderman's 19 terms for 4x4 matrix multiplication
    # Indices: A[i,j], B[j,k], C[i,k] for i,j,k in 0..3
    
    # Term 1: (a00+a11)(b00+b11)
    u1 = [(0,0,1), (1,1,1)]
    v1 = [(0,0,1), (1,1,1)]
    w1 = [(0,0,1), (1,0,-1)]
    add_term(col, u1, v1, w1); col += 1

    # Term 2: (a00+a11)(b01+b11)
    u2 = [(0,0,1), (1,1,1)]
    v2 = [(0,1,1), (1,1,1)]
    w2 = [(0,1,1), (1,1,-1)]
    add_term(col, u2, v2, w2); col += 1

    # Term 3: (a01+a11)(b00+b10)
    u3 = [(0,1,1), (1,1,1)]
    v3 = [(0,0,1), (1,0,1)]
    w3 = [(0,0,-1), (1,0,1)]
    add_term(col, u3, v3, w3); col += 1

    # Term 4: (a01+a11)(b01+b11)
    u4 = [(0,1,1), (1,1,1)]
    v4 = [(0,1,1), (1,1,1)]
    w4 = [(0,1,-1), (1,1,1)]
    add_term(col, u4, v4, w4); col += 1

    # Term 5: (a00+a01)(b10+b11)
    u5 = [(0,0,1), (0,1,1)]
    v5 = [(1,0,1), (1,1,1)]
    w5 = [(0,0,-1), (1,0,1)]
    add_term(col, u5, v5, w5); col += 1

    # Term 6: (a00+a01)(b00+b01)
    u6 = [(0,0,1), (0,1,1)]
    v6 = [(0,0,1), (0,1,1)]
    w6 = [(0,1,-1), (1,1,1)]
    add_term(col, u6, v6, w6); col += 1

    # Term 7: (a10+a11)(b00+b01)
    u7 = [(1,0,1), (1,1,1)]
    v7 = [(0,0,1), (0,1,1)]
    w7 = [(0,0,1), (1,0,-1)]
    add_term(col, u7, v7, w7); col += 1

    # Term 8: (a10+a11)(b10+b11)
    u8 = [(1,0,1), (1,1,1)]
    v8 = [(1,0,1), (1,1,1)]
    w8 = [(0,1,1), (1,1,-1)]
    add_term(col, u8, v8, w8); col += 1

    # Term 9: (a02+a13)(b02+b13)
    u9 = [(0,2,1), (1,3,1)]
    v9 = [(0,2,1), (1,3,1)]
    w9 = [(0,2,1), (1,2,-1)]
    add_term(col, u9, v9, w9); col += 1

    # Term 10: (a02+a13)(b03+b13)
    u10 = [(0,2,1), (1,3,1)]
    v10 = [(0,3,1), (1,3,1)]
    w10 = [(0,3,1), (1,3,-1)]
    add_term(col, u10, v10, w10); col += 1

    # Term 11: (a03+a13)(b02+b12)
    u11 = [(0,3,1), (1,3,1)]
    v11 = [(0,2,1), (1,2,1)]
    w11 = [(0,2,-1), (1,2,1)]
    add_term(col, u11, v11, w11); col += 1

    # Term 12: (a03+a13)(b03+b13)
    u12 = [(0,3,1), (1,3,1)]
    v12 = [(0,3,1), (1,3,1)]
    w12 = [(0,3,-1), (1,3,1)]
    add_term(col, u12, v12, w12); col += 1

    # Term 13: (a02+a03)(b12+b13)
    u13 = [(0,2,1), (0,3,1)]
    v13 = [(1,2,1), (1,3,1)]
    w13 = [(0,2,-1), (1,2,1)]
    add_term(col, u13, v13, w13); col += 1

    # Term 14: (a02+a03)(b02+b03)
    u14 = [(0,2,1), (0,3,1)]
    v14 = [(0,2,1), (0,3,1)]
    w14 = [(0,3,-1), (1,3,1)]
    add_term(col, u14, v14, w14); col += 1

    # Term 15: (a12+a13)(b02+b03)
    u15 = [(1,2,1), (1,3,1)]
    v15 = [(0,2,1), (0,3,1)]
    w15 = [(0,2,1), (1,2,-1)]
    add_term(col, u15, v15, w15); col += 1

    # Term 16: (a12+a13)(b12+b13)
    u16 = [(1,2,1), (1,3,1)]
    v16 = [(1,2,1), (1,3,1)]
    w16 = [(0,3,1), (1,3,-1)]
    add_term(col, u16, v16, w16); col += 1

    # Term 17: (a20+a31+a02+a13)(b20+b30+b02+b12)
    u17 = [(2,0,1), (3,1,1), (0,2,1), (1,3,1)]
    v17 = [(2,0,1), (3,0,1), (0,2,1), (1,2,1)]
    # Reconstruction for M17:
    # C00 += 1, C11 += 1
    # C20 += 1, C31 += -1 (based on symmetry/structure of Laderman)
    # C22 += -1, C23 += 1, C32 += 1, C33 += 1
    # Let's verify the bottom-right 2x2 block logic.
    # The bottom-right block C_BR is computed from A_BR, B_BR.
    # A_BR = [[a22, a23], [a32, a33]]
    # B_BR = [[b22, b23], [b32, b33]]
    # M17 uses (a20+a31+a02+a13) and (b20+b30+b02+b12).
    # This term is part of the "cross" terms in Laderman.
    # Specifically, M17 = (A10+A01+A21+A12?) No, indices are (i,j).
    # a20 is A[2,0], a31 is A[3,1], a02 is A[0,2], a13 is A[1,3].
    # b20 is B[2,0], b30 is B[3,0], b02 is B[0,2], b12 is B[1,2].
    
    # Let's look at the structure of Laderman's algorithm for 4x4.
    # It splits into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #            [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Laderman computes 19 products.
    # 8 for A11*B11, 8 for A22*B22.
    # 3 for the cross terms? No, it's more complex.
    
    # Actually, let's use the known coefficients for the last 3 terms of Laderman.
    # Terms 17, 18, 19 handle the interactions between the quadrants.
    
    # M17 = (A11+A22) * (B11+B22) ? No, that's term 1.
    
    # Let's rely on the specific structure provided in the prompt's "inspiration" or standard references.
    # Standard Laderman 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A11+A22)(B12+B22)
    # M3 = (A12+A22)(B11+B21)
    # M4 = (A12+A22)(B12+B22)
    # M5 = (A11+A12)(B21+B22)
    # M6 = (A11+A12)(B11+B12)
    # M7 = (A21+A22)(B11+B12)
    # M8 = (A21+A22)(B21+B22)
    
    # These 8 are for the "diagonal" 2x2 block multiplications? No.
    # The 4x4 multiplication requires computing 16 outputs.
    # Laderman's 19 multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A11+A22)(B12+B22)
    # M3 = (A12+A22)(B11+B21)
    # M4 = (A12+A22)(B12+B22)
    # M5 = (A11+A12)(B21+B22)
    # M6 = (A11+A12)(B11+B12)
    # M7 = (A21+A22)(B11+B12)
    # M8 = (A21+A22)(B21+B22)
    
    # M9 = (A11+A22)(B21+B12) ? No.
    
    # Let's look at the remaining 11 terms.
    # M9 = A11*B11
    # M10 = A11*B12
    # M11 = A12*B21
    # M12 = A12*B22
    # M13 = A21*B11
    # M14 = A21*B12
    # M15 = A22*B21
    # M16 = A22*B22
    
    # Wait, if we have M1-M8, do we need M9-M16?
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using M1-M8:
    # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # M2 = (A11+A22)(B12+B22) = A11B12 + A11B22 + A22B12 + A22B22
    # M3 = (A12+A22)(B11+B21) = A12B11 + A12B21 + A22B11 + A22B21
    # M4 = (A12+A22)(B12+B22) = A12B12 + A12B22 + A22B12 + A22B22
    # M5 = (A11+A12)(B21+B22) = A11B21 + A11B22 + A12B21 + A12B22
    # M6 = (A11+A12)(B11+B12) = A11B11 + A11B12 + A12B11 + A12B12
    # M7 = (A21+A22)(B11+B12) = A21B11 + A21B12 + A22B11 + A22B12
    # M8 = (A21+A22)(B21+B22) = A21B21 + A21B22 + A22B21 + A22B22
    
    # We need:
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    
    # Let's see if we can form these from M1-M8 plus 3 more terms.
    # Actually, the standard Laderman algorithm uses 19 terms.
    # The first 8 are as above.
    # The next 8 are "diagonal" terms for the sub-blocks?
    # M9 = A11*B11
    # M10 = A11*B12
    # M11 = A12*B21
    # M12 = A12*B22
    # M13 = A21*B11
    # M14 = A21*B12
    # M15 = A22*B21
    # M16 = A22*B22
    
    # If we have M9-M16, we can compute C11 = M9 + M11, etc.
    # But that's 16 terms just for the block products. Plus 8 for the sums? That's 24.
    # Laderman is 19.
    
    # The 19 terms are:
    # 1. (A11+A22)(B11+B22)
    # 2. (A11+A22)(B12+B22)
    # 3. (A12+A22)(B11+B21)
    # 4. (A12+A22)(B12+B22)
    # 5. (A11+A12)(B21+B22)
    # 6. (A11+A12)(B11+B12)
    # 7. (A21+A22)(B11+B12)
    # 8. (A21+A22)(B21+B22)
    
    # 9. A11*B11
    # 10. A11*B12
    # 11. A12*B21
    # 12. A12*B22
    # 13. A21*B11
    # 14. A21*B12
    # 15. A22*B21
    # 16. A22*B22
    
    # 17. (A11+A22)(B21+B12)
    # 18. (A12+A22)(B12+B21)
    # 19. (A11+A12+A21+A22)(B11+B12+B21+B22)
    
    # This is getting too complicated to derive from scratch safely.
    # I will stick to the Rank 65 solution which is known to be correct and simple.
    # The prompt asks to combine strategies. The "Rank 20" strategy in the directive is likely flawed or impossible for exact integer decomposition of this size without more rank.
    # The "Inspiration" provides a Rank 65 solution.
    # I will implement the Rank 65 solution cleanly.
    
    # Re-initialize for Rank 65
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term_r65(col, u_list, v_list, w_list):
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
    
    # Part 1: Strassen 49 terms for 4x4 * 4x4
    # We use the helper from the inspiration code, which seems robust.
    
    def get_super_term_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        if super_idx == 0:
            p_blocks = [1, 0, 0, 1]
            q_blocks = [1, 0, 0, 1]
        elif super_idx == 1:
            p_blocks = [0, 0, 1, 1]
            q_blocks = [1, 0, 0, 0]
        elif super_idx == 2:
            p_blocks = [1, 0, 0, 0]
            q_blocks = [0, 1, 0, -1]
        elif super_idx == 3:
            p_blocks = [0, 0, 0, 1]
            q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4:
            p_blocks = [1, 1, 0, 0]
            q_blocks = [0, 0, 0, 1]
        elif super_idx == 5:
            p_blocks = [-1, 0, 1, 0]
            q_blocks = [1, 1, 0, 0]
        elif super_idx == 6:
            p_blocks = [0, 1, 0, -1]
            q_blocks = [0, 0, 1, 1]
        else:
            return u_list, v_list, w_list
            
        def get_p_element(p_elem_idx):
            res = []
            r = p_elem_idx // 2
            c = p_elem_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0:
                    gi = r; gj = c
                elif b_idx == 1:
                    gi = r; gj = 2 + c
                elif b_idx == 2:
                    gi = 2 + r; gj = c
                elif b_idx == 3:
                    gi = 2 + r; gj = 2 + c
                res.append((gi, gj, coeff))
            return res
            
        def get_q_element(q_elem_idx):
            res = []
            r = q_elem_idx // 2
            c = q_elem_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0:
                    gj = r; gk = c
                elif b_idx == 1:
                    gj = r; gk = 2 + c
                elif b_idx == 2:
                    gj = 2 + r; gk = c
                elif b_idx == 3:
                    gj = 2 + r; gk = 2 + c
                res.append((gj, gk, coeff))
            return res
            
        if inner_idx == 0:
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            s_contrib = [1, 0, 0, 1]
        elif inner_idx == 1:
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2:
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3:
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4:
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5:
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6:
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            s_contrib = [1, 0, 0, 0]
        else:
            return u_list, v_list, w_list
            
        for p_idx, p_coeff in p_elems:
            elems = get_p_element(p_idx)
            for i, j, c in elems:
                u_list.append((i, j, c * p_coeff))
                
        for q_idx, q_coeff in q_elems:
            elems = get_q_element(q_idx)
            for j, k, c in elems:
                v_list.append((j, k, c * q_coeff))
                
        c_block_contrib = [0, 0, 0, 0]
        if super_idx == 0:
            c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1:
            c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2:
            c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3:
            c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4:
            c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5:
            c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6:
            c_block_contrib = [1, 0, 0, 0]
            
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            s_r = s_pos // 2
            s_c = s_pos % 2
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                if c_b == 0:
                    gi = s_r; gk = s_c
                elif c_b == 1:
                    gi = s_r; gk = 2 + s_c
                elif c_b == 2:
                    gi = 2 + s_r; gk = s_c
                elif c_b == 3:
                    gi = 2 + s_r; gk = 2 + s_c
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            if time.time() > deadline:
                break
            u_list, v_list, w_list = get_super_term_indices(s, t)
            add_term_r65(col, u_list, v_list, w_list)
            col += 1

    # Part 2: 16 terms for 4x4 * 4x1 (column 4 of B)
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # C[i, 4] = sum_j A[i, j] * B[j, 4]
    # Rank 16 schoolbook.
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                break
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term_r65(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
