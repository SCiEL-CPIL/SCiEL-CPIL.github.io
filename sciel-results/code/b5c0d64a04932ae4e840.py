import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Construct rank-28 decomposition using Strassen blocks
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x6) into B1 (2x6) and B2 (2x6)
    # C = A1*B1 + A2*B2
    # Each 2x2 @ 2x6 is decomposed using Strassen on three 2x2 blocks
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (24, 28)
    W = [[0] * rank for _ in range(p * n)]  # (12, 28)
    
    # Strassen 2x2 @ 2x2 decomposition
    # For A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
    # M2 = (a10+a11)b00 -> C10 += M2
    # M3 = a00(b01-b11) -> C01 += M3
    # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
    # M5 = (a00+a01)b11 -> C01 += M5, C00 -= M5
    # M6 = (a10-a00)(b00+b01) -> C11 += M6
    # M7 = (a01-a11)(b10+b11) -> C00 += M7, C11 -= M7
    
    # Define the 7 Strassen products as (U_vec, V_vec, W_vec) for 2x2 @ 2x2
    # U_vec is in R^4 (for a00,a01,a10,a11)
    # V_vec is in R^4 (for b00,b01,b10,b11)
    # W_vec is in R^4 (for C00,C01,C10,C11)
    strassen_products = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 1, 1, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]), # M7
    ]
    
    col = 0
    # For A1*B1: A1 is rows 0-1, cols 0-1 of A. B1 is rows 0-1, cols 0-5 of B.
    # B1 is 2x6, split into three 2x2 blocks: cols 0-1, 2-3, 4-5
    for block in range(3):
        for (u_vec, v_vec, w_vec) in strassen_products:
            # U: A1 entries are at positions (0,0),(0,1),(1,0),(1,1) in A
            # A is 2x4, so A[i,j] is at index i*4+j
            A_indices = [0*4+0, 0*4+1, 1*4+0, 1*4+1]  # a00,a01,a10,a11
            for idx, val in zip(A_indices, u_vec):
                U[idx][col] += val
            
            # V: B1 entries are at positions (0,0),(0,1),(1,0),(1,1) of B1
            # B1 is 2x6, so B1[i,j] is at index i*6+j in B
            # Block 'block' uses cols block*2, block*2+1
            B_indices = [0*6+block*2, 0*6+block*2+1, 1*6+block*2, 1*6+block*2+1]
            for idx, val in zip(B_indices, v_vec):
                V[idx][col] += val
            
            # W: C entries. C is 2x6, C[i,k] at index k*2+i
            # The 2x2 block of C for this block is at cols block*2, block*2+1
            # C00 -> C[0, block*2] -> index block*2*2+0
            # C01 -> C[0, block*2+1] -> index (block*2+1)*2+0
            # C10 -> C[1, block*2] -> index block*2*2+1
            # C11 -> C[1, block*2+1] -> index (block*2+1)*2+1
            C_indices = [block*2*2+0, (block*2+1)*2+0, block*2*2+1, (block*2+1)*2+1]
            for idx, val in zip(C_indices, w_vec):
                W[idx][col] += val
            
            col += 1
    
    # For A2*B2: A2 is rows 0-1, cols 2-3 of A. B2 is rows 2-3, cols 0-5 of B.
    for block in range(3):
        for (u_vec, v_vec, w_vec) in strassen_products:
            # U: A2 entries are at positions (0,2),(0,3),(1,2),(1,3) in A
            A_indices = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
            for idx, val in zip(A_indices, u_vec):
                U[idx][col] += val
            
            # V: B2 entries are at positions (0,0),(0,1),(1,0),(1,1) of B2
            # B2 is 2x6, rows 2-3 of B, so B2[i,j] is at index (2+i)*6+j in B
            B_indices = [2*6+block*2, 2*6+block*2+1, 3*6+block*2, 3*6+block*2+1]
            for idx, val in zip(B_indices, v_vec):
                V[idx][col] += val
            
            # W: same as before
            C_indices = [block*2*2+0, (block*2+1)*2+0, block*2*2+1, (block*2+1)*2+1]
            for idx, val in zip(C_indices, w_vec):
                W[idx][col] += val
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

