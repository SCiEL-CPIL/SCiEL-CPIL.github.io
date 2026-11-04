import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We'll build the decomposition by composing Strassen decompositions.
    # For the 4x4 * 4x4 part, we use two levels of Strassen: 49 rank.
    # For the 4x4 * 4x2 part, we use one level of Strassen: 28 rank.
    # Total rank: 77.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps
    # Each step is (alpha, beta, gamma) where each is a dict from (i,j) to coeff
    S7 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B, first 4 columns of C)
    # Two-level Strassen: 7 outer steps x 7 inner steps = 49 columns
    for outer in range(7):
        alpha_o, beta_o, gamma_o = S7[outer]
        for inner in range(7):
            alpha_i, beta_i, gamma_i = S7[inner]
            col = outer * 7 + inner
            
            # U: A is 4x4, index (row, col) -> row * m + col
            for (bo, co), vo in alpha_o.items():
                for (bi, ci), vi in alpha_i.items():
                    r = (bo * 2 + bi) * m + (co * 2 + ci)
                    U[r][col] += vo * vi
            
            # V: B is 4x4 (first 4 cols), index (row, col) -> row * p + col
            for (bo, co), vo in beta_o.items():
                for (bi, ci), vi in beta_i.items():
                    r = (bo * 2 + bi) * p + (co * 2 + ci)
                    V[r][col] += vo * vi
            
            # W: C is 4x4 (first 4 cols), index (col, row) -> col * n + row
            for (bo, co), vo in gamma_o.items():
                for (bi, ci), vi in gamma_i.items():
                    r = (co * 2 + ci) * n + (bo * 2 + bi)
                    W[r][col] += vo * vi
    
    # Part 2: 4x4 * 4x2 (last 2 columns of B, last 2 columns of C)
    # A is 4x4, B is 4x2, C is 4x2
    # We can think of this as: A = [A00 A01; A10 A11], B = [B0; B1] where B0, B1 are 2x2
    # C = [C0; C1] where C0, C1 are 2x2
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # We need 4 block multiplications: A00*B0, A01*B1, A10*B0, A11*B1
    # Each is 2x2 * 2x2 -> 2x2, rank 7
    # Total: 4 * 7 = 28 columns
    
    col_offset = 49
    # (a_block_row, a_block_col, b_block_row) -> c_block_row
    # a_block: (0,0) is top-left 2x2 of A
    # b_block: 0 is top 2x2 of B, 1 is bottom 2x2 of B
    # c_block: 0 is top 2x2 of C, 1 is bottom 2x2 of C
    
    mults = [
        # (a_row, a_col, b_row, c_row)
        (0, 0, 0, 0),  # A00 * B0 -> C0
        (0, 1, 1, 0),  # A01 * B1 -> C0
        (1, 0, 0, 1),  # A10 * B0 -> C1
        (1, 1, 1, 1),  # A11 * B1 -> C1
    ]
    
    for m_idx, (ar, ac, br, cr) in enumerate(mults):
        for inner in range(7):
            alpha_i, beta_i, gamma_i = S7[inner]
            col = col_offset + m_idx * 7 + inner
            
            # U: A block at (ar, ac), 2x2
            for (bi, ci), vi in alpha_i.items():
                r = (ar * 2 + bi) * m + (ac * 2 + ci)
                U[r][col] += vi
            
            # V: B block at row br, 2x2, cols 4-5
            for (bi, ci), vi in beta_i.items():
                r = (br * 2 + bi) * p + (4 + ci)
                V[r][col] += vi
            
            # W: C block at row cr, 2x2, cols 4-5
            for (bi, ci), vi in gamma_i.items():
                r = (4 + ci) * n + (cr * 2 + bi)
                W[r][col] += vi
    
    # Verification pass (optional, for debugging)
    # We can verify the decomposition by checking a few random matrices
    # But this is expensive, so we skip it in production
    
    return {"U": U, "V": V, "W": W}
