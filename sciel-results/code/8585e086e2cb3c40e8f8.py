import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 4x4x4 using block Strassen (7 products x 8 schoolbook terms = 56)
    # 4x4x3 using schoolbook (4*4*3 = 48)
    total_rank = 56 + 48
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen 2x2x2 block definitions
    # Each: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from (block_row, block_col) to coeff for A
    # b_coeffs: dict from (block_row, block_col) to coeff for B
    # c_coeffs: dict from (block_row, block_col) to coeff for C
    strassen_blocks = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # M3 = A00(B10-B00)
        ({(0,0): 1}, {(1,0): 1, (0,0): -1},
         {(0,1): 1, (1,1): 1}),
        # M4 = A11(B01-B11)
        ({(1,1): 1}, {(0,1): 1, (1,1): -1},
         {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    # 2x2x2 schoolbook: 8 terms
    # For 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # Terms: (i,j,k) for i,j,k in {0,1}
    schoolbook_2x2 = [(i,j,k) for i in range(2) for j in range(2) for k in range(2)]
    
    # Part 1: 4x4x4 (columns 0-3 of B)
    for s_idx in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_blocks[s_idx]
        for (si, sj, sk) in schoolbook_2x2:
            r = current_rank
            current_rank += 1
            
            # For each (si,sj,sk), the term is:
            # sum over a_blocks, b_blocks, c_blocks:
            # a_coeff[a_block] * b_coeff[b_block] * c_coeff[c_block] * A[a_block*2+si, a_block*2+sj] * B[b_block*2+sj, b_block*2+sk] * C[c_block*2+si, c_block*2+sk]
            
            # Wait, this isn't right. The Strassen product is a scalar, and it contributes to multiple C blocks.
            
            # Actually, each Strassen product M_s is a scalar (product of two linear combinations of 2x2 blocks).
            # But that's not right either - M_s is a 2x2 matrix product of two linear combinations.
            
            # I think the issue is that the Strassen algorithm works at the block level, where each "element" is a 2x2 matrix.
            # So M1 = (A00+A11)(B00+B11) is a 2x2 matrix product.
            # The 2x2 matrix product has rank 8 (schoolbook) or 7 (Strassen).
            # If I use schoolbook within, it's 8 rank-1 terms.
            
            # For each rank-1 term in the 2x2 schoolbook, I need to expand the linear combinations.
            
            # This is getting very complex. Let me just use the full schoolbook for now.
            pass
    
    # Given the complexity, let me just use the full schoolbook
    current_rank = 0
    U = [[0.0] * 112 for _ in range(n * m)]
    V = [[0.0] * 112 for _ in range(m * p)]
    W = [[0.0] * 112 for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = current_rank
                current_rank += 1
                U[i*m + j][r] = 1.0
                V[j*p + k][r] = 1.0
                W[k*n + i][r] = 1.0
    
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
