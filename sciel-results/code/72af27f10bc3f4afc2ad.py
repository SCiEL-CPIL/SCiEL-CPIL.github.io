import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Rank 36 construction
    rank = 36
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        nonlocal col
        U[u_idx][col] = val_u
        V[v_idx][col] = val_v
        W[w_idx][col] = val_w
        col += 1

    # Block 1: B cols 0-3
    # A11: A[0:2, 0:2]
    # A12: A[0:2, 2:4]
    
    # M1 = A11 * (B11 + B22)
    # B11: B[0:2, 0:2], B22: B[2:4, 2:4]
    # Terms:
    # A11[0,0] * (B[0,0] + B[2,2]) -> 2 terms
    # A11[0,0] * (B[0,1] + B[2,3]) -> 2 terms
    # A11[0,1] * (B[1,0] + B[3,2]) -> 2 terms
    # A11[0,1] * (B[1,1] + B[3,3]) -> 2 terms
    # A11[1,0] * (B[0,0] + B[2,2]) -> 2 terms
    # ...
    
    # This is tedious. I will write a loop.
    
    # Define blocks
    # A11 rows 0,1 cols 0,1
    # A12 rows 0,1 cols 2,3
    # B11 rows 0,1 cols 0,1
    # B12 rows 0,1 cols 2,3
    # B21 rows 2,3 cols 0,1
    # B22 rows 2,3 cols 2,3
    
    # M1 = A11 * (B11 + B22)
    # M3 = A11 * (B12 - B22)
    # M5 = (A11 + A12) * B22
    # M6 = -A11 * (B11 + B12)
    # M7 = A12 * (B21 + B22)
    
    # We will generate terms for these.
    
    # Function to add product of A_block (2x2) and B_block_sum (list of (block, sign))
    # A_block is defined by (row_start, col_start)
    # B_block is defined by (row_start, col_start)
    
    def add_product(a_r, a_c, b_blocks, w_r, w_c):
        # a_r, a_c: start row, col of A block (2x2)
        # b_blocks: list of (b_r, b_c, sign)
        # w_r, w_c: start row, col of C block (2x2)
        nonlocal col
        
        for i in range(2):
            for j in range(2):
                # C[i,j] += sum_k A[i,k] * B[k,j]
                # We decompose into terms for each k and each B block
                for k in range(2):
                    # A[i, a_c+k]
                    u_idx = (a_r + i) * 4 + (a_c + k)
                    val_u = 1
                    
                    for b_r, b_c, sign in b_blocks:
                        # B[b_r+k, b_c+j]
                        v_idx = (b_r + k) * 7 + (b_c + j)
                        val_v = sign
                        
                        # C[w_r+i, w_c+j]
                        w_idx = (w_c + j) * 2 + (w_r + i)
                        val_w = 1
                        
                        add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w)

    # M1
    add_product(0, 0, [(0, 0, 1), (2, 2, 1)], 0, 0)
    # M3
    add_product(0, 0, [(0, 2, 1), (2, 2, -1)], 0, 2)
    # M5
    # (A11 + A12) * B22
    # A11 + A12 is not a single block.
    # A11: (0,0), A12: (0,2)
    # We can call add_product for A11 and A12 separately?
    # No, add_product assumes a single A block.
    # We can modify add_product to take a list of A blocks.
    
    # Let's refactor add_product to take list of A blocks.
    
    def add_product_multi(a_blocks, b_blocks, w_r, w_c):
        nonlocal col
        for a_r, a_c, sign_a in a_blocks:
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u_idx = (a_r + i) * 4 + (a_c + k)
                        val_u = sign_a
                        for b_r, b_c, sign_b in b_blocks:
                            v_idx = (b_r + k) * 7 + (b_c + j)
                            val_v = sign_b
                            w_idx = (w_c + j) * 2 + (w_r + i)
                            val_w = 1
                            add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w)

    # M1
    add_product_multi([(0, 0, 1)], [(0, 0, 1), (2, 2, 1)], 0, 0)
    # M3
    add_product_multi([(0, 0, 1)], [(0, 2, 1), (2, 2, -1)], 0, 2)
    # M5
    add_product_multi([(0, 0, 1), (0, 2, 1)], [(2, 2, 1)], 0, 2)
    # M6
    add_product_multi([(0, 0, -1)], [(0, 0, 1), (0, 2, 1)], 0, 0)
    # M7
    add_product_multi([(0, 2, 1)], [(2, 0, 1), (2, 2, 1)], 0, 0)
    
    # This covers the first 4 columns of B.
    # Now for the next 3 columns (4,5,6).
    # Pad to 4 columns (4,5,6,7).
    # B11: (0,4), B12: (0,6) -- wait, B12 is 2x2.
    # Cols 4,5,6.
    # B11: cols 4,5. B12: cols 6,7(0).
    # B21: cols 4,5. B22: cols 6,7(0).
    
    # M1
    add_product_multi([(0, 0, 1)], [(0, 4, 1), (2, 6, 1)], 0, 4)
    # M3
    add_product_multi([(0, 0, 1)], [(0, 6, 1), (2, 6, -1)], 0, 6)
    # M5
    add_product_multi([(0, 0, 1), (0, 2, 1)], [(2, 6, 1)], 0, 6)
    # M6
    add_product_multi([(0, 0, -1)], [(0, 4, 1), (0, 6, 1)], 0, 4)
    # M7
    add_product_multi([(0, 2, 1)], [(2, 4, 1), (2, 6, 1)], 0, 4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
