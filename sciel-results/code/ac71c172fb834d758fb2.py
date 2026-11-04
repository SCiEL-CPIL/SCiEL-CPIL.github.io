import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split B (4x8) into B1 (4x4) and B2 (4x4)
    # C (4x8) into C1 (4x4) and C2 (4x4)
    # We compute C1 = A B1 and C2 = A B2 using rank 28 decompositions each.
    
    rank_per_part = 28
    total_rank = 2 * rank_per_part
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # idx_u: index in U (0..15)
    # idx_v: index in V (0..31)
    # idx_w: index in W (0..31)
    # col: column in U, V, W
    # val: coefficient
    def add_term(idx_u, idx_v, idx_w, col, val):
        U[idx_u][col] += val
        V[idx_v][col] += val
        W[idx_w][col] += val
        
    # Define Strassen multipliers for 4x4x4
    # Each multiplier k defines linear combos of 2x2 blocks.
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Strassen Multipliers (coeffs for A, B, C)
    # k: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {block_idx: coeff}
    # block_idx: 0=00, 1=01, 2=10, 3=11
    
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3 = A00(B01-B11) -> C01
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        # M4 = A11(B10-B00) -> C10, C00
        ({3: 1}, {2: 1, 0: -1}, {2: 1, 0: -1}),
        # M5 = (A00+A01)B11 -> C01, C00
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
        # M6 = (A10-A00)(B00+B01) -> C11
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00, C11
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1})
    ]
    
    # Schoolbook terms for 2x2x2
    # s: (a_row, a_col, b_row, b_col, c_row, c_col)
    # A[i,k] * B[k,j] -> C[i,j]
    schoolbook_terms = [
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 1, 0),
        (1, 1, 1, 1, 1, 1)
    ]
    
    # Function to get global index for a 2x2 block element
    # block_idx: 0(00), 1(01), 2(10), 3(11)
    # r, c: row, col within 2x2 block (0 or 1)
    # returns (row, col) in 4x4
    def block_to_idx(block_idx, r, c):
        br = block_idx // 2
        bc = block_idx % 2
        return br * 2 + r, bc * 2 + c

    # Process Part 1 (B1, C1)
    # B1 corresponds to cols 0..3 of B
    # C1 corresponds to cols 0..3 of C
    for k in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_defs[k]
        for s in range(4):
            a_r, a_c, b_r, b_c, c_r, c_c = schoolbook_terms[s]
            
            # We need to sum over all blocks that contribute to A[a_r, a_c] and B[b_r, b_c]
            # Wait, the Strassen definition gives coefficients for the *blocks*.
            # So A_block[br, bc] has coefficient coeff.
            # The element A[br*2+a_r, bc*2+a_c] is part of A_block[br, bc].
            
            # So for a given schoolbook term (a_r, a_c, b_r, b_c, c_r, c_c):
            # We iterate over all A-blocks (ab) with non-zero coeff
            # and all B-blocks (bb) with non-zero coeff
            # and all C-blocks (cb) with non-zero coeff?
            # No, the C-block is determined by the Strassen formula.
            # The Strassen formula says C_block[cr, cc] += coeff_c * M_k.
            # M_k is the result of the block multiplication.
            # The schoolbook term computes one part of M_k.
            # Specifically, it computes contribution to M_k[c_r, c_c] from A_block[a_r, a_c] * B_block[b_r, b_c].
            # This contribution goes to C_block[c_r, c_c] scaled by coeff_c?
            # No.
            # M_k is a 2x2 matrix.
            # C_block[cr, cc] = sum_k coeff_c[k, cr, cc] * M_k[cr, cc].
            # So for a fixed k, M_k[cr, cc] contributes to C_block[cr, cc] with weight coeff_c[k, cr, cc].
            
            # So, for each k:
            #   For each schoolbook term s (which defines a_r, a_c, b_r, b_c, c_r, c_c in the 2x2 block space):
            #     For each A-block ab with coeff ca:
            #       For each B-block bb with coeff cb:
            #         The term is ca * cb * (A_elem * B_elem).
            #         This contributes to M_k[c_r, c_c].
            #         M_k[c_r, c_c] contributes to C_block[c_r, c_c] with weight cc_coeff (from c_coeffs dict).
            #         So total weight = ca * cb * cc_coeff.
            #         The global indices are determined by ab, a_r, a_c for U; bb, b_r, b_c for V; c_r, c_c for W?
            #         Wait, W is indexed by C. The C-block is determined by c_r, c_c in the Strassen formula?
            #         No, the Strassen formula maps M_k to C blocks.
            #         c_coeffs dict tells us which C blocks are affected by M_k.
            #         But M_k is a 2x2 matrix. Its elements M_k[0,0], M_k[0,1], etc.
            #         The schoolbook term s computes M_k[c_r, c_c].
            #         So the C-block affected is the one at position (c_r, c_c) in the 2x2 block grid?
            #         NO. The Strassen formula is:
            #         C_00 = M_1 + M_4 - M_5 + M_7
            #         C_01 = M_3 + M_5
            #         C_10 = M_2 + M_4
            #         C_11 = M_1 - M_2 + M_3 + M_6
            #         Here C_00, C_01 etc are 2x2 blocks.
            #         So M_1 (a 2x2 matrix) is added to C_00 (a 2x2 matrix) and C_11 (a 2x2 matrix).
            #         This means M_1[0,0] adds to C_00[0,0] AND C_11[0,0].
            #         M_1[0,1] adds to C_00[0,1] AND C_11[0,1].
            #         
            #         So for a given k, and a given schoolbook term s (which produces M_k[cr, cc]):
            #         We iterate over all C-blocks cb that are in c_coeffs.
            #         For each such cb, the element M_k[cr, cc] is added to C_block[cb][cr, cc].
            #         
            #         So:
            #         for k in 0..6:
            #           for s in 0..3:
            #             a_r, a_c, b_r, b_c, m_c_r, m_c_c = schoolbook_terms[s]
            #             for ab in a_coeffs:
            #               for bb in b_coeffs:
            #                 for cb in c_coeffs:
            #                   weight = a_coeffs[ab] * b_coeffs[bb] * c_coeffs[cb]
            #                   idx_u = block_to_idx(ab, a_r, a_c)
            #                   idx_v = block_to_idx(bb, b_r, b_c)
            #                   idx_w = block_to_idx(cb, m_c_r, m_c_c)
            #                   add_term(idx_u, idx_v, idx_w, current_col, weight)
            #                   current_col += 1

    # Wait, the column index needs to be incremented for every rank-1 term.
    # The loop structure above generates the terms.
    
    col = 0
    for part in range(2):
        # part 0: B1, C1 (cols 0-3)
        # part 1: B2, C2 (cols 4-7)
        # The U matrix is the same for both parts (A is shared).
        # V and W indices shift.
        
        for k in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_defs[k]
            for s in range(4):
                a_r, a_c, b_r, b_c, m_c_r, m_c_c = schoolbook_terms[s]
                for ab in a_coeffs:
                    for bb in b_coeffs:
                        for cb in c_coeffs:
                            weight = a_coeffs[ab] * b_coeffs[bb] * c_coeffs[cb]
                            if weight == 0: continue
                            
                            u_row, u_col = block_to_idx(ab, a_r, a_c)
                            u_idx = u_row * 4 + u_col
                            
                            # V index
                            # B is 4x8.
                            # Block bb is 2x2.
                            # b_r, b_c are within the 2x2 block.
                            # Global B row = br * 2 + b_r
                            # Global B col = bc * 2 + b_c + part * 4
                            v_row, v_col = block_to_idx(bb, b_r, b_c)
                            v_idx = v_row * 8 + (v_col + part * 4)
                            
                            # W index
                            # C is 4x8.
                            # Block cb is 2x2.
                            # m_c_r, m_c_c are within the 2x2 block.
                            # Global C row = cr * 2 + m_c_r
                            # Global C col = cc * 2 + m_c_c + part * 4
                            w_row, w_col = block_to_idx(cb, m_c_r, m_c_c)
                            w_idx = w_row * 8 + (w_col + part * 4)
                            
                            U[u_idx][col] += weight
                            V[v_idx][col] += weight
                            W[w_idx][col] += weight
                            
                            col += 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
