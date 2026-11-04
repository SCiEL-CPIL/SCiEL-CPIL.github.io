"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1
                
    # Rank-22 Decomposition
    # Strategy:
    # 1. Split the 3x3 interaction (middle part of A and B) into:
    #    - A_top (2x2) * B_top (2x4)  -> This is effectively two 2x2 multiplications
    #    - A_bottom (2x1) * B_bottom (1x4) -> This is 8 scalar multiplications
    # 2. For A_top * B_top:
    #    - A_top is rows 0,1 of A, cols 0,1.
    #    - B_top is rows 0,1 of B, cols 0,1 and cols 2,3.
    #    - We can treat B_top as two 2x2 blocks: B_top_left (cols 0,1) and B_top_right (cols 2,3).
    #    - Each 2x2 * 2x2 multiplication can be done in rank 7 (Strassen).
    #    - So 2 * 7 = 14 rank.
    # 3. For A_bottom * B_bottom:
    #    - A_bottom is rows 0,1 of A, col 2.
    #    - B_bottom is row 2 of B, cols 0,1,2,3.
    #    - This is a sum of outer products, rank 2 * 4 = 8.
    # Total Rank = 14 + 8 = 22.
    
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    # Helper functions to accumulate values
    def add_u(i, j, r, val):
        U22[i * m + j, r] += val
    def add_v(j, k, r, val):
        V22[j * p + k, r] += val
    def add_w(k, i, r, val):
        W22[k * n + i, r] += val
        
    # --- Part 1: A_bottom * B_bottom (Rank 8) ---
    # A_bottom: A[0,2], A[1,2]
    # B_bottom: B[2,0], B[2,1], B[2,2], B[2,3]
    # C contribution: C[i,k] += A[i,2] * B[2,k]
    r_base = 0
    for i in range(n): # i in 0, 1
        for k in range(p): # k in 0, 1, 2, 3
            r = r_base + i * p + k
            add_u(i, 2, r, 1)
            add_v(2, k, r, 1)
            add_w(k, i, r, 1)
            
    # --- Part 2: A_top * B_top (Rank 14) ---
    # A_top: A[0,0], A[0,1], A[1,0], A[1,1]
    # B_top_left: B[0,0], B[0,1], B[1,0], B[1,1] -> contributes to C[:, 0:2]
    # B_top_right: B[0,2], B[0,3], B[1,2], B[1,3] -> contributes to C[:, 2:4]
    
    # We use Strassen's algorithm for 2x2 matrix multiplication.
    # Let A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]].
    # The 7 products are:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b10)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # Reconstruction:
    # C00 = m1 - m5 + m7 + m4
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    def add_strassen_block(b_col_offset, c_col_offset, r_base):
        # b_col_offset: 0 for left block (cols 0,1), 2 for right block (cols 2,3)
        # c_col_offset: 0 for left block (cols 0,1), 2 for right block (cols 2,3)
        
        # A indices are fixed:
        # a00: (0,0)
        # a01: (0,1)
        # a10: (1,0)
        # a11: (1,1)
        
        # B indices depend on b_col_offset
        # b00: (0, b_col_offset)
        # b01: (0, b_col_offset+1)
        # b10: (1, b_col_offset)
        # b11: (1, b_col_offset+1)
        
        # C indices depend on c_col_offset
        # c00: (c_col_offset, 0) -> k=c_col_offset, i=0
        # c01: (c_col_offset+1, 0) -> k=c_col_offset+1, i=0
        # c10: (c_col_offset, 1) -> k=c_col_offset, i=1
        # c11: (c_col_offset+1, 1) -> k=c_col_offset+1, i=1
        
        # m1
        r = r_base + 0
        add_u(0, 0, r, 1)
        add_u(1, 1, r, 1)
        add_v(0, b_col_offset, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, 1)    # C00
        add_w(c_col_offset+1, 1, r, 1)  # C11
        
        # m2
        r = r_base + 1
        add_u(1, 0, r, 1)
        add_u(1, 1, r, 1)
        add_v(0, b_col_offset, r, 1)
        add_w(c_col_offset, 1, r, 1)    # C10
        add_w(c_col_offset+1, 1, r, -1) # C11
        
        # m3
        r = r_base + 2
        add_u(0, 0, r, 1)
        add_v(0, b_col_offset+1, r, 1)
        add_v(1, b_col_offset+1, r, -1)
        add_w(c_col_offset+1, 0, r, 1)  # C01
        add_w(c_col_offset+1, 1, r, 1)  # C11
        
        # m4
        r = r_base + 3
        add_u(1, 1, r, 1)
        add_v(1, b_col_offset, r, 1)
        add_v(0, b_col_offset, r, -1)
        add_w(c_col_offset, 0, r, 1)    # C00
        add_w(c_col_offset, 1, r, 1)    # C10
        
        # m5
        r = r_base + 4
        add_u(0, 0, r, 1)
        add_u(0, 1, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, -1)   # C00
        add_w(c_col_offset+1, 0, r, 1)  # C01
        
        # m6
        r = r_base + 5
        add_u(1, 0, r, 1)
        add_u(0, 0, r, -1)
        add_v(0, b_col_offset, r, 1)
        add_v(1, b_col_offset, r, 1)
        add_w(c_col_offset+1, 1, r, 1)  # C11
        
        # m7
        r = r_base + 6
        add_u(0, 1, r, 1)
        add_u(1, 1, r, -1)
        add_v(1, b_col_offset, r, 1)
        add_v(1, b_col_offset+1, r, 1)
        add_w(c_col_offset, 0, r, 1)    # C00
        
    # Block 1: Left part (cols 0,1)
    add_strassen_block(0, 0, 8)
    
    # Block 2: Right part (cols 2,3)
    add_strassen_block(2, 2, 15)
    
    # Verification
    def verify_decomp(U, V, W, rank):
        # T[k, i, j] should be 1 if C[k,i] = sum A[i,j]B[j,k]
        # Check all k, i, j
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True
        
    if verify_decomp(U22, V22, W22, rank_22):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
        
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
