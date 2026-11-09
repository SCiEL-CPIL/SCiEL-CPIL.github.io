"""
Tensor decomposition for 4x5 by 5x5 matrix multiplication using Strassen's algorithm
on 2x2 sub-blocks where applicable, with trivial decomposition for remaining blocks.

The output factor follows the official trace convention W[k*n+i, r].
U has shape (n*m, rank) = (20, rank)
V has shape (m*p, rank) = (25, rank)
W has shape (p*n, rank) = (20, rank)
"""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We'll build up the decomposition by processing blocks.
    # Split the 5 dimension into 2+3.
    # A is 4x5, B is 5x5, C is 4x5.
    # 
    # Split rows of A: i in [0,2) and i in [2,4)
    # Split columns of A / rows of B: j in [0,2) and j in [2,5)
    # Split columns of B: k in [0,2) and k in [2,5)
    #
    # This gives 8 blocks:
    # Block (i_block, j_block, k_block) where:
    #   i_block in {0,1} (rows 0-1, rows 2-3)
    #   j_block in {0,1} (cols 0-1, cols 2-4)
    #   k_block in {0,1} (cols 0-1, cols 2-4)
    #
    # Block sizes:
    #   i_block 0: 2 rows, i_block 1: 2 rows
    #   j_block 0: 2 cols, j_block 1: 3 cols
    #   k_block 0: 2 cols, k_block 1: 3 cols
    #
    # For blocks where all three dimensions are 2x2x2, we can use Strassen (rank 7).
    # Otherwise, we use trivial decomposition.
    
    # Strassen's 7 multiplications for 2x2 matrices:
    # C00 = M1 - M5 + M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U entries for A (row-major: A00, A01, A10, A11)
    # V entries for B (row-major: B00, B01, B10, B11)
    # W entries for C (row-major: C00, C01, C10, C11)
    
    strassen_terms = [
        # (U coeffs, V coeffs, W coeffs)
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (A10+A11)B00 -> C10 += 1
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3: A00(B10-B11) -> C01 += 1
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        # M4: A11(B01-B00) -> C00 += 1, C10 += 1
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (A00+A01)B11 -> C00 -= 1, C10 += 1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Build the full decomposition
    # We'll accumulate contributions into U, V, W lists
    
    # First, determine total rank
    # Count ranks for each block
    rank = 0
    for i_blk in range(2):
        for j_blk in range(2):
            for k_blk in range(2):
                i_size = 2  # always 2 for 4x...
                j_size = 2 if j_blk == 0 else 3
                k_size = 2 if k_blk == 0 else 3
                if i_size == 2 and j_size == 2 and k_size == 2:
                    rank += 7  # Strassen
                else:
                    rank += i_size * j_size * k_size  # trivial
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 20 x rank
    V = [[0] * rank for _ in range(m * p)]  # 25 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    col = 0
    for i_blk in range(2):
        for j_blk in range(2):
            for k_blk in range(2):
                i_start = i_blk * 2
                i_end = i_start + 2
                j_start = 0 if j_blk == 0 else 2
                j_end = 2 if j_blk == 0 else 5
                k_start = 0 if k_blk == 0 else 2
                k_end = 2 if k_blk == 0 else 5
                
                i_size = i_end - i_start  # always 2
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for (a_idx, u_val) in u_coeffs.items():
                            # a_idx: 0=A00, 1=A01, 2=A10, 3=A11
                            # Row in A: a_idx // 2, Col in A: a_idx % 2
                            a_row = a_idx // 2
                            a_col = a_idx % 2
                            # Global indices
                            global_i = i_start + a_row
                            global_j = j_start + a_col
                            U[global_i * m + global_j][col] = u_val
                        
                        for (b_idx, v_val) in v_coeffs.items():
                            # b_idx: 0=B00, 1=B01, 2=B10, 3=B11
                            b_row = b_idx // 2
                            b_col = b_idx % 2
                            global_j = j_start + b_row
                            global_k = k_start + b_col
                            V[global_j * p + global_k][col] = v_val
                        
                        for (c_idx, w_val) in w_coeffs.items():
                            # c_idx: 0=C00, 1=C01, 2=C10, 3=C11
                            c_row = c_idx // 2
                            c_col = c_idx % 2
                            global_i = i_start + c_row
                            global_k = k_start + c_col
                            # W convention: W[k*n + i, r]
                            W[global_k * n + global_i][col] = w_val
                        
                        col += 1
                else:
                    # Trivial decomposition
                    for i in range(i_size):
                        for j in range(j_size):
                            for k in range(k_size):
                                global_i = i_start + i
                                global_j = j_start + j
                                global_k = k_start + k
                                U[global_i * m + global_j][col] = 1
                                V[global_j * p + global_k][col] = 1
                                W[global_k * n + global_i][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
