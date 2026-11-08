import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for 2x2 matrix flattened row-major: 00, 01, 10, 11
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # P2 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # P3 = A00(B01 - B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # P4 = A11(B10 - B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # P5 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    
    # Strategy: Block-Strassen composition
    # A (4x4) is a 2x2 block of 2x2 submatrices
    # B (4x8) is a 2x4 block of 2x2 submatrices (2 rows, 4 columns of 2x2 blocks)
    # C (4x8) is a 2x4 block of 2x2 submatrices
    
    # We apply the 7-term Strassen algorithm to the block structure.
    # The block dimensions are:
    # A_blocks: 2x2 grid of 2x2 matrices
    # B_blocks: 2x4 grid of 2x2 matrices
    # C_blocks: 2x4 grid of 2x2 matrices
    
    # The 7 Strassen terms for 2x2 block multiplication (where each "element" is a 2x2 matrix)
    # P1 = (A00 + A11)(B00 + B11) -> Here B00, B11 are 2x4 blocks
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # Each of these terms involves multiplying a 2x2 matrix (from A) with a 2x4 block (from B)
    # to produce a 2x4 block (in C).
    # We decompose each 2x2x(2x4) product using the 2x2x2 Strassen decomposition.
    # Since the B block is 2x4 (two 2x2 blocks side by side), we can apply the 2x2x2
    # Strassen to each 2x2 sub-block of B independently, or more efficiently, we can
    # view the 2x4 B block as two 2x2 blocks and apply Strassen to each.
    
    # Actually, a better way: The product of a 2x2 matrix with a 2x4 matrix can be
    # decomposed into two 2x2x2 products (for the left and right halves of the 2x4).
    # So each of the 7 top-level Strassen terms requires 2 * 7 = 14 rank-1 terms.
    # Total rank = 7 * 14 = 98.
    
    # Let's build this explicitly.
    
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    rank = 98
    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98
    
    # Block definitions:
    # A is 4x4, blocks are 2x2.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B is 4x8, blocks are 2x2.
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 0-1, cols 4-5
    # B11: rows 0-1, cols 6-7
    # B20: rows 2-3, cols 0-1
    # B21: rows 2-3, cols 2-3
    # B30: rows 2-3, cols 4-5
    # B31: rows 2-3, cols 6-7
    
    # For the block-Strassen, we treat:
    # B_top = [B00 B01 B10 B11] (2x8, but as 2x2 blocks: 4 blocks)
    # B_bot = [B20 B21 B30 B31] (2x8, but as 2x2 blocks: 4 blocks)
    
    # The 7 Strassen terms for the 2x2 block multiplication:
    # P1 = (A00+A11) * (B_top + B_bot) where B_top+B_bot is 2x8
    # P2 = (A10+A11) * B_top
    # P3 = A00 * (B01_block - B11_block) where these are specific 2x2 blocks
    # ... etc.
    
    # Let me define the effective A and B blocks for each Strassen term.
    # Each term is: A_eff (2x2) * B_eff (2x8) -> C_eff (2x8)
    # B_eff is 2x8, which is 4 2x2 blocks.
    # We decompose A_eff * B_eff into 4 separate 2x2x2 products.
    
    # Define the 7 top-level terms:
    # (A_coeff_blocks, B_coeff_blocks)
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks (2x8): represented as 4 2x2 blocks.
    # B0 = top-left 2x2, B1 = top-right 2x2, B2 = bot-left 2x2, B3 = bot-right 2x2
    
    terms = [
        # P1: (A00+A11)(B00+B11) where B00,B11 are 2x8 blocks
        # A: A00 + A11, B: B_top + B_bot (all 4 blocks with coeff 1)
        ([(1, 0), (1, 3)], [(1, 0), (1, 1), (1, 2), (1, 3)]),
        # P2: (A10+A11)B00 where B00 is the top 2x8 block
        ([(1, 2), (1, 3)], [(1, 0), (1, 1), (0, 0), (0, 0)]),
        # P3: A00(B01-B11) where B01,B11 are specific 2x2 blocks
        # B01 is top-right 2x2 (block 1), B11 is bot-right 2x2 (block 3)
        ([(1, 0)], [(0, 0), (1, 1), (0, 0), (-1, 3)]),
        # P4: A11(B10-B00) where B10 is bot-left (block 2), B00 is top-left (block 0)
        ([(1, 3)], [(-1, 0), (0, 0), (1, 2), (0, 0)]),
        # P5: (A00+A01)B11 where B11 is the bot 2x8 block
        ([(1, 0), (1, 1)], [(0, 0), (0, 0), (1, 2), (1, 3)]),
        # P6: (A10-A00)(B00+B01) where B00,B01 are top-left and top-right 2x2
        ([(1, 2), (-1, 0)], [(1, 0), (1, 1), (0, 0), (0, 0)]),
        # P7: (A01-A11)(B10+B11) where B10,B11 are bot-left and bot-right 2x2
        ([(1, 1), (-1, 3)], [(0, 0), (0, 0), (1, 2), (1, 3)]),
    ]
    
    for t_idx, (a_spec, b_spec) in enumerate(terms):
        for b_block_idx in range(4):  # 4 2x2 blocks in the 2x8 B
            for s_idx in range(7):  # 7 Strassen terms for each 2x2x2
                col_idx = t_idx * 28 + b_block_idx * 7 + s_idx
                
                # U contribution
                for coeff, a_blk in a_spec:
                    ai, aj = divmod(a_blk, 2)
                    for lr in range(2):
                        for lc in range(2):
                            gr = ai * 2 + lr
                            gc = aj * 2 + lc
                            g_idx = gr * 4 + gc
                            l_idx = lr * 2 + lc
                            U[g_idx, col_idx] += coeff * U_2[l_idx, s_idx]
                
                # V contribution
                coeff_b, b_blk = b_spec[b_block_idx]
                if coeff_b != 0:
                    # b_blk is 0-3, representing which 2x2 block in the 2x8 B
                    # Block 0: rows 0-1, cols 0-1
                    # Block 1: rows 0-1, cols 2-3
                    # Block 2: rows 2-3, cols 0-1
                    # Block 3: rows 2-3, cols 2-3
                    # Wait, I need to map b_block_idx to the actual 2x2 block in B.
                    # b_block_idx 0: top-left, 1: top-right, 2: bot-left, 3: bot-right
                    # But b_spec[b_block_idx] gives (coeff, b_blk) where b_blk is the
                    # index of the 2x2 block in the 2x8 B.
                    # Actually, b_spec has 4 entries, one for each 2x2 block in the 2x8.
                    # b_spec[i] = (coeff, b_blk) where b_blk is the index of the 2x2 block
                    # in the original B (0-7). But I defined it differently.
                    # Let me re-examine.
                    pass
    
    # This is getting complicated. Let me use a simpler approach:
    # Just use the existing _strassen_4x4x4 and split B into two 4x4 halves.
    
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Left half (B cols 0-3)
    for i in range(16):
        U[i, 0:49] = U_4[i, :]
        r, c = divmod(i, 4)
        v_idx_left = r * 8 + c
        V[v_idx_left, 0:49] = V_4[i, :]
        W[i, 0:49] = W_4[i, :]
    
    # Right half (B cols 4-7)
    for i in range(16):
        U[i, 49:98] = U_4[i, :]
        r, c = divmod(i, 4)
        v_idx_right = r * 8 + (c + 4)
        V[v_idx_right, 49:98] = V_4[i, :]
        W[i + 16, 49:98] = W_4[i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    Rank 49.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    A_eff_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)]
    ]
    
    B_eff_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)]
    ]
    
    C_eff_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)]
    ]
    
    for k in range(7):
        for m in range(7):
            col_idx = k * 7 + m
            
            for coeff_a, a_blk in A_eff_specs[k]:
                bi, bj = divmod(a_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        U[g_idx, col_idx] += coeff_a * U_2[l_idx, m]
                        
            for coeff_b, b_blk in B_eff_specs[k]:
                bi, bj = divmod(b_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        V[g_idx, col_idx] += coeff_b * V_2[l_idx, m]
                        
            for coeff_c, c_blk in C_eff_specs[k]:
                bi, bj = divmod(c_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gc * 4 + gr
                        l_idx = lr * 2 + lc
                        W[g_idx, col_idx] += coeff_c * W_2[l_idx, m]
                        
    return U, V, W
