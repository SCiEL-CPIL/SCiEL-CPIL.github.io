"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-77 exact tensor decomposition for 4x4 by 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    rank = 77
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0

    # Strassen steps for 2x2x2
    # Each step is (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict { (i,j): coeff }
    # B_coeffs: dict { (i,j): coeff }
    # C_coeffs: dict { (i,j): coeff }
    
    def strassen_step(idx):
        if idx == 0:
            return ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
        elif idx == 1:
            return ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1})
        elif idx == 2:
            return ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
        elif idx == 3:
            return ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1})
        elif idx == 4:
            return ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1})
        elif idx == 5:
            return ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
        elif idx == 6:
            return ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        return None

    # Helper to get U index for A block (bi, bj) and local (i, j)
    # A is 4x4. Blocks are 2x2.
    # Block (bi, bj) covers A[bi*2:i, bj*2:j]
    def u_idx(bi, bj, i, j):
        return (bi * 2 + i) * m + (bj * 2 + j)

    # Helper to get V index for B block
    # B is 4x6.
    # For B1 (cols 0-3): Blocks (bi, bj) where bj in 0,1. Cols bj*2 + j.
    # For B2 (cols 4-5): Blocks (bi, bj) where bj in 0,1. Cols 4 + bj*2 + j.
    def v_idx_b1(bi, bj, i, j):
        return (bi * 2 + i) * p + (bj * 2 + j)
        
    def v_idx_b2(bi, bj, i, j):
        return (bi * 2 + i) * p + (4 + bj * 2 + j)

    # Helper to get W index for C block
    # C is 6x4. W[k*n+i]. k is col in C (0-5), i is row in C (0-3).
    # For C1 (cols 0-3): Blocks (bi, bj) where bi is row block (0,1), bj is col block (0,1).
    # C[bi*2+i, bj*2+j]. W index: (bj*2+j)*n + (bi*2+i)
    def w_idx_c1(bi, bj, i, j):
        return (bj * 2 + j) * n + (bi * 2 + i)
        
    # For C2 (cols 4-5): Blocks (bi, bj) where bi is row block (0,1), bj is col block (0,1).
    # C[bi*2+i, 4+bj*2+j]. W index: (4+bj*2+j)*n + (bi*2+i)
    def w_idx_c2(bi, bj, i, j):
        return (4 + bj * 2 + j) * n + (bi * 2 + i)

    # Part 1: C1 = A * B1 (4x4 * 4x4), Rank 49
    # Outer Strassen on 4x4 * 4x4
    # A blocks: A00, A01, A10, A11
    # B1 blocks: B1_00, B1_01, B1_10, B1_11
    # C1 blocks: C1_00, C1_01, C1_10, C1_11
    
    for s_outer in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_step(s_outer)
        # For each outer step, we form A_combo (2x2) and B_combo (2x2)
        # A_combo[i,j] = sum_{ba} a_coeffs[ba] * A[ba][i,j]
        # B_combo[i,j] = sum_{bb} b_coeffs[bb] * B1[bb][i,j]
        # C_combo[i,j] = sum_{bc} c_coeffs[bc] * C1[bc][i,j]
        
        # Inner Strassen on A_combo * B_combo
        for s_inner in range(7):
            ai_coeffs, bi_coeffs, ci_coeffs = strassen_step(s_inner)
            
            # Accumulate U, V, W for this rank-1 term
            # U:
            for (ba_i, ba_j), va in a_coeffs.items():
                for (ai_i, ai_j), vi in ai_coeffs.items():
                    # A[ba][ai, aj]
                    U[u_idx(ba_i, ba_j, ai_i, ai_j), col] += va * vi
            
            # V:
            for (bb_i, bb_j), vb in b_coeffs.items():
                for (bi_i, bi_j), vi in bi_coeffs.items():
                    # B1[bb][bi, bj]
                    V[v_idx_b1(bb_i, bb_j, bi_i, bi_j), col] += vb * vi
            
            # W:
            for (bc_i, bc_j), vc in c_coeffs.items():
                for (ci_i, ci_j), vi in ci_coeffs.items():
                    # C1[bc][ci, cj]
                    W[w_idx_c1(bc_i, bc_j, ci_i, ci_j), col] += vc * vi
            
            col += 1

    # Part 2: C2 = A * B2 (4x4 * 4x2), Rank 28
    # B2 is 4x2. Split into B2_top (2x2) and B2_bot (2x2).
    # B2_top: rows 0-1, cols 4-5.
    # B2_bot: rows 2-3, cols 4-5.
    
    # C2 is 4x2. Split into C2_top (2x2) and C2_bot (2x2).
    # C2_top: rows 0-1, cols 4-5.
    # C2_bot: rows 2-3, cols 4-5.
    
    # C2_top = A00 * B2_top + A01 * B2_bot
    # C2_bot = A10 * B2_top + A11 * B2_bot
    
    # We compute 4 Strassen products:
    # 1. A00 * B2_top -> add to C2_top
    # 2. A01 * B2_bot -> add to C2_top
    # 3. A10 * B2_top -> add to C2_bot
    # 4. A11 * B2_bot -> add to C2_bot
    
    products = [
        # (A_block_idx, B_block_type, C_block_idx)
        # A_block_idx: (bi, bj) for A
        # B_block_type: 'top' or 'bot'
        # C_block_idx: (bi, bj) for C
        ((0, 0), 'top', (0, 0)), # A00 * B2_top -> C2_top
        ((0, 1), 'bot', (0, 0)), # A01 * B2_bot -> C2_top
        ((1, 0), 'top', (1, 0)), # A10 * B2_top -> C2_bot
        ((1, 1), 'bot', (1, 0)), # A11 * B2_bot -> C2_bot
    ]
    
    for (a_bi, a_bj), b_type, (c_bi, c_bj) in products:
        for s in range(7):
            ai_coeffs, bi_coeffs, ci_coeffs = strassen_step(s)
            
            # U: from A[a_bi, a_bj]
            for (ai_i, ai_j), va in ai_coeffs.items():
                U[u_idx(a_bi, a_bj, ai_i, ai_j), col] += va
            
            # V: from B2 block
            # B2_top: bi=0, bj in 0,1. V idx: v_idx_b2(0, bj, i, j)
            # B2_bot: bi=1, bj in 0,1. V idx: v_idx_b2(1, bj, i, j)
            b_bi = 0 if b_type == 'top' else 1
            for (bi_i, bi_j), vb in bi_coeffs.items():
                V[v_idx_b2(b_bi, bi_j, bi_i, bi_j), col] += vb
            
            # W: from C2 block
            # C2_top: c_bi=0, c_bj in 0,1. W idx: w_idx_c2(0, c_bj, i, j)
            # C2_bot: c_bi=1, c_bj in 0,1. W idx: w_idx_c2(1, c_bj, i, j)
            # Note: c_bj in products list is always 0? 
            # C2_top is 2x2. Its blocks are (0,0) and (0,1) in terms of local 2x2?
            # No, C2_top is the whole 2x2 block.
            # In the Strassen step, we are multiplying 2x2 * 2x2.
            # The result is a 2x2 matrix.
            # This 2x2 matrix corresponds to C2_top (if c_bi=0) or C2_bot (if c_bi=1).
            # So c_bj in the product definition refers to the column block of the 2x2 result?
            # No, the Strassen step produces a full 2x2 result.
            # So we add the full 2x2 result to C2_top or C2_bot.
            # The W indices for C2_top are w_idx_c2(0, bj, i, j) for bj in 0,1.
            # The W indices for C2_bot are w_idx_c2(1, bj, i, j) for bj in 0,1.
            
            for (ci_i, ci_j), vc in ci_coeffs.items():
                W[w_idx_c2(c_bi, ci_j, ci_i, ci_j), col] += vc
            
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
