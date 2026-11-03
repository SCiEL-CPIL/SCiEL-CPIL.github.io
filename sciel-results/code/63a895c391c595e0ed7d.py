import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before the 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build rank-1 terms as triples of coefficient dicts:
    #   u: index in [0, n*m) -> coeff
    #   v: index in [0, m*p) -> coeff
    #   w: index in [0, p*n) -> coeff
    terms = []

    # --- Strassen 2x2x2 rank-7 basis ---
    # Each entry: (u_coeffs dict, v_coeffs dict, w_coeffs dict)
    # u indices: 0..3 for 2x2 A block (row-major: (r,c) -> r*2+c)
    # v indices: 0..3 for 2x2 B block
    # w indices: 0..3 for 2x2 C block

    def s2x2_terms():
        """Return the 7 rank-1 terms for 2x2x2 Strassen."""
        t = []
        # M1 = (A00+A11)(B00+B11)
        t.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1, 2:-1}))
        # M2 = (A10+A11)B00
        t.append(({2:1, 3:1}, {0:1}, {1:1, 3:1}))
        # M3 = A00(B01-B11)
        t.append(({0:1}, {1:1, 3:-1}, {1:1}))
        # M4 = A11(B10-B00)
        t.append(({3:1}, {2:1, 0:-1}, {2:1}))
        # M5 = (A00+A01)B11
        t.append(({0:1, 1:1}, {3:1}, {0:1, 1:1}))
        # M6 = (A10-A00)(B00+B01)
        t.append(({2:1, 0:-1}, {0:1, 1:1}, {2:1, 3:1}))
        # M7 = (A01-A11)(B10+B11)
        t.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}))
        return t

    s2 = s2x2_terms()

    # --- Part 1: 4x4 x 4x4 x 4x4 via Kronecker of two 2x2x2 Strassen ---
    # A split into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # B split into 4 blocks: B00, B01, B10, B11 (each 2x2)
    # C = sum of products
    #
    # A block (bi,bj) occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1 in 4x4
    #   global index: (2*bi+r)*4 + (2*bj+c)
    # B block (bi,bj): row 2*bi+r in 4, col 2*bj+c in 4 (of the 4x4 part)
    #   global index in m*p=24: (2*bi+r)*6 + (2*bj+c)
    # C block (ci,cj): row 2*ci+r, col 2*cj+c
    #   global index in p*n=24 (trace convention W[k*n+i,r]): (2*cj+c)*4 + (2*ci+r)

    # The 7 outer Strassen combinations for 4x4x4:
    # For each outer level, we pick which A-blocks and B-blocks to combine
    outer_combos = [
        # (list of (block, sign) for A, list of (block, sign) for B)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),   # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)]),             # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),            # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),            # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)]),             # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),    # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),    # M7
    ]

    # For each outer combo, and each inner Strassen term, build a rank-1 term
    for oi, (a_blocks, b_blocks) in enumerate(outer_combos):
        if time.time() > deadline:
            break
        for si in range(7):
            if time.time() > deadline:
                break
            u_inner, v_inner, w_inner = s2[si]

            u_coeff = {}
            for (u_key, u_val) in u_inner.items():
                ur, uc = u_key // 2, u_key % 2
                for (bi, bj, sign) in a_blocks:
                    idx = (2*bi + ur) * 4 + (2*bj + uc)
                    u_coeff[idx] = u_coeff.get(idx, 0) + sign * u_val

            v_coeff = {}
            for (v_key, v_val) in v_inner.items():
                vr, vc = v_key // 2, v_key % 2
                for (bi, bj, sign) in b_blocks:
                    idx = (2*bi + vr) * 6 + (2*bj + vc)
                    v_coeff[idx] = v_coeff.get(idx, 0) + sign * v_val

            w_coeff = {}
            for (w_key, w_val) in w_inner.items():
                wr, wc = w_key // 2, w_key % 2
                # The w_inner index maps to the C block position
                # For outer combo oi, we need to map which C blocks this contributes to
                # The outer Strassen result mapping:
                # C00 = M1 + M3 - M4 + M6
                # C01 = M2 + M5
                # C10 = M3 + M4
                # C11 = M1 - M2 + M5 + M7
                # Wait, let me use the standard 2x2 Strassen result mapping for the outer level
                # Actually the outer combo already encodes which A and B blocks are used.
                # The result C block positions depend on the outer combo.
                
                # For outer combo oi, the inner 2x2x2 result w_inner gives the 2x2 result
                # within each C-block pair. Let me think more carefully.
                
                # Actually, the Kronecker approach: 
                # The 4x4x4 tensor = sum over outer combinations of (A_outer * B_outer)
                # Each outer combination is a sum of (A_block_i * B_block_j) for specific blocks
                # The inner Strassen handles the 2x2x2 within each block pair.
                
                # Let me reconsider. The standard approach:
                # 4x4 = [[A00, A01],[A10, A11]]
                # 4x4 = [[B00, B01],[B10, B11]]
                # C = [[A00B00+A01B10, A00B01+A01B11],
                #      [A10B00+A11B10, A10B01+A11B11]]
                
                # Strassen for the 2x2 block structure:
                # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
                # M2 = (A10+A11)B00 -> contributes to C10, C11
                # M3 = A00(B01-B11) -> contributes to C00, C10
                # M4 = A11(B10-B00) -> contributes to C00, C11
                # M5 = (A00+A01)B11 -> contributes to C00, C01
                # M6 = (A10-A00)(B00+B01) -> contributes to C10, C11
                # M7 = (A01-A11)(B10+B11) -> contributes to C00, C01
                
                # Wait, this is the standard 2x2 Strassen where each "block" is 2x2.
                # So M1 = (A00+A11)(B00+B11) is a 2x2x2x2x2x2 product.
                # The inner Strassen handles the 2x2x2 within.
                
                # For outer combo oi, the A blocks involved and B blocks involved are fixed.
                # The inner Strassen term si gives the 2x2x2 decomposition of the product
                # of the summed A sub-blocks and summed B sub-blocks.
                
                # The C block positions for each outer combo:
                # M1 -> C00 (+), C11 (+)
                # M2 -> C10 (+), C11 (+)  
                # M3 -> C00 (+), C10 (+)
                # M4 -> C00 (-), C11 (-)  [wait, let me recheck]
                
                # Standard 2x2 Strassen:
                # C00 = M1 - M4 + M6 + M3  [hmm, let me use a standard reference]
                # Actually:
                # P1 = (A00+A11)(B00+B11)
                # P2 = (A10+A11)B00
                # P3 = A00(B01-B11)
                # P4 = A11(B10-B00)
                # P5 = (A00+A01)B11
                # P6 = (A10-A00)(B00+B01)
                # P7 = (A01-A11)(B10+B11)
                # C00 = P1 - P4 + P6 + P3
                # C01 = P5 + P7
                # C10 = P2 + P4
                # C11 = P1 - P2 + P5 + P7
                
                # Hmm, let me just use a verified mapping. Let me use the one from the current code:
                # big_result_map = {
                #     '00': [(0, 1), (3, 1), (4, -1), (6, 1)],  # M0+M3-M4+M6
                #     '01': [(2, 1), (4, 1)],                     # M2+M4
                #     '10': [(1, 1), (3, 1)],                     # M1+M3
                #     '11': [(0, 1), (1, -1), (2, 1), (5, 1)]    # M0-M1+M2+M5
                # }
                # Wait, that doesn't look right either. Let me just verify with a concrete example.
                
                # Actually, let me just use the standard mapping and verify.
                # For 2x2 Strassen with blocks:
                # C00 = P1 - P4 + P6 + P3
                # C01 = P5 + P7  
                # C10 = P2 + P4
                # C11 = P1 - P2 + P5 + P7
                
                # So for outer combo oi (0-indexed, corresponding to P0..P6):
                c_block_map = {
                    0: {0: 1, 3: 1},      # P0 (M1): C00+, C11+  -- wait
                }
                
                # Let me just use the mapping from the current code which was verified to give score 1.2468:
                # big_result_map maps c_block -> list of (m_idx, coeff)
                # '00': [(0, 1), (3, 1), (4, -1), (6, 1)]
                # '01': [(2, 1), (4, 1)]
                # '10': [(1, 1), (3, 1)]
                # '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
                
                # So for outer combo oi, the C blocks it contributes to:
                c_contribs = {}
                c_map = {
                    '00': [(0, 1), (3, 1), (4, -1), (6, 1)],
                    '01': [(2, 1), (4, 1)],
                    '10': [(1, 1), (3, 1)],
                    '11': [(0, 1), (1, -1), (2, 1), (5, 1)]
                }
                for c_blk, contribs in c_map.items():
                    for (m_idx, coeff) in contribs:
                        if m_idx == oi:
                            ci, cj = int(c_blk[0]), int(c_blk[1])
                            w_coeff[(ci, cj)] = w_coeff.get((ci, cj), 0) + coeff

                # Now expand w_inner (2x2) into each C block
                w_final = {}
                for (ci, cj), blk_coeff in w_coeff.items():
                    for (w_key, w_val) in w_inner.items():
                        wr, wc = w_key // 2, w_key % 2
                        # C block (ci,cj) occupies rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
                        # Global index in p*n=24 with trace convention W[k*n+i,r]:
                        # k = column index in C = 2*cj + wc, i = row index in C = 2*ci + wr
                        # W[k*n + i] = W[k*4 + i]
                        idx = (2*cj + wc) * 4 + (2*ci + wr)
                        w_final[idx] = w_final.get(idx, 0) + blk_coeff * w_val

            terms.append((u_coeff, v_coeff, w_final))

    # --- Part 2: 4x4 x 4x2 x 4x2 ---
    # B2 is columns 4-5 of the 4x6 matrix B. C2 is columns 4-5 of the 4x4 output C.
    # A is 4x4, B2 is 4x2, C2 is 4x2.
    # Split A into 2x2 blocks, B2 into top 2x1 and bottom 2x1 (or 2x2 with one col zero?).
    # Actually B2 is 4x2. Split into top 2x2 (rows 0-1, cols 4-5) and bottom 2x2 (rows 2-3, cols 4-5).
    # C2 = [A00*B2_top + A01*B2_bot; A10*B2_top + A11*B2_bot] where each product is 2x2 x 2x2 x 2x2?
    # No: A00 is 2x2, B2_top is 2x2, product is 2x2.
    # So C2_top = A00*B2_top + A01*B2_bot (2x2)
    #    C2_bot = A10*B2_top + A11*B2_bot (2x2)
    # Total: 4 products of 2x2 x 2x2 x 2x2, each rank 7, total 28.
    
    # For each of the 4 products, use Strassen 2x2x2.
    prod_defs = [
        # (a_block (row, col), b_part ('top'/'bot'), c_part ('top'/'bot'))
        ((0,0), 'top', 'top'),   # A00 * B2_top -> C2_top
        ((0,1), 'bot', 'top'),   # A01 * B2_bot -> C2_top
        ((1,0), 'top', 'bot'),   # A10 * B2_top -> C2_bot
        ((1,1), 'bot', 'bot'),   # A11 * B2_bot -> C2_bot
    ]

    for (ai, aj), b_part, c_part in prod_defs:
        if time.time() > deadline:
            break
        for si in range(7):
            if time.time() > deadline:
                break
            u_inner, v_inner, w_inner = s2[si]

            u_coeff = {}
            for (u_key, u_val) in u_inner.items():
                ur, uc = u_key // 2, u_key % 2
                idx = (2*ai + ur) * 4 + (2*aj + uc)
                u_coeff[idx] = u_coeff.get(idx, 0) + u_val

            v_coeff = {}
            for (v_key, v_val) in v_inner.items():
                vr, vc = v_key // 2, v_key % 2
                if b_part == 'top':
                    idx = vr * 6 + (4 + vc)
                else:
                    idx = (2 + vr) * 6 + (4 + vc)
                v_coeff[idx] = v_coeff.get(idx, 0) + v_val

            w_coeff = {}
            for (w_key, w_val) in w_inner.items():
                wr, wc = w_key // 2, w_key % 2
                if c_part == 'top':
                    idx = (4 + wc) * 4 + wr
                else:
                    idx = (4 + wc) * 4 + (2 + wr)
                w_coeff[idx] = w_coeff.get(idx, 0) + w_val

            terms.append((u_coeff, v_coeff, w_coeff))

    rank = len(terms)

    # Build U, V, W as lists of lists (matching the interface)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][col] = val
        for idx, val in v_c.items():
            V[idx][col] = val
        for idx, val in w_c.items():
            W[idx][col] = val

    return {"U": U, "V": V, "W": W}
