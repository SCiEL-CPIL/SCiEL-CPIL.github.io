"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication
    using the tensor product of Strassen (2x2, rank 7) and schoolbook (3x3, rank 27).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_i is a 3x3 matrix (since 6/2 = 3)
    # Each M_i = X_i * Y_i where X_i, Y_i are 3x3 matrices
    
    # For the tensor decomposition, each scalar multiplication in the 3x3 products
    # contributes rank-1 terms
    
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    
    # Define Strassen's 7 products
    # Each product has:
    # - A-side: list of (block_row, block_col, coefficient)
    # - B-side: list of (block_row, block_col, coefficient)
    # - Output: which output block(s) this contributes to, and with what sign
    
    strassen_products = [
        # (a_blocks, b_blocks, [(out_block_row, out_block_col, sign), ...])
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),      # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),                 # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),                 # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),                 # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),                 # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),                 # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),                 # M7
    ]
    
    block_size = 3  # 6 / 2 = 3
    
    for a_blocks, b_blocks, out_blocks in strassen_products:
        # X = sum of a_blocks (3x3 matrix)
        # Y = sum of b_blocks (3x3 matrix)
        # M = X * Y (3x3 matrix multiplication)
        
        # For each element (r, c) of M:
        for r in range(block_size):
            for c in range(block_size):
                # M[r,c] = sum_k X[r,k] * Y[k,c]
                for k in range(block_size):
                    # X[r,k] = sum_{(br, bc, ca)} ca * A[3*br+r, 3*bc+k]
                    # Y[k,c] = sum_{(br, bc, cb)} cb * B[3*br+k, 3*bc+c]
                    
                    # For each combination:
                    for (abr, abc, ca) in a_blocks:
                        for (bbr, bbc, cb) in b_blocks:
                            # A element: (3*abr + r, 3*abc + k)
                            # B element: (3*bbr + k, 3*bbc + c)
                            
                            # For valid matrix mult, need 3*abc + k == 3*bbr + k
                            # => abc == bbr
                            
                            if abc == bbr:
                                a_i = 3 * abr + r
                                a_j = 3 * abc + k
                                b_j = 3 * bbr + k
                                b_k = 3 * bbc + c
                                
                                # This term contributes to output blocks
                                # The output element is at (3*obr + r, 3*obc + c)
                                # for each (obr, obc, sign) in out_blocks
                                
                                coeff = ca * cb
                                if coeff == 0:
                                    continue
                                
                                # For each output block this contributes to:
                                for (obr, obc, sign) in out_blocks:
                                    out_i = 3 * obr + r
                                    out_j = 3 * obc + c
                                    
                                    # Rank-1 term: A[a_i, a_j] * B[b_j, b_k]
                                    # contributes sign * coeff to C[out_i, out_j]
                                    
                                    # U[a_i * m + a_j, col] = ca
                                    # V[b_j * p + b_k, col] = cb
                                    # W[out_j * n + out_i, col] = sign
                                    
                                    U[a_i * m + a_j][col] = ca
                                    V[b_j * p + b_k][col] = cb
                                    W[out_j * n + out_i][col] = sign
                                    col += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
