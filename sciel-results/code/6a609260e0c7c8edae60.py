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
    
    # Build the exact multiplication tensor T
    # T[k, i, j] is the coefficient of A[i,j] * B[j,k] in C[i,k]
    # Flattened indices:
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # We want U, V, W such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[k, i, j]
    
    # Schoolbook decomposition (Rank 24)
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
    
    # Try to find a lower rank decomposition using Strassen-like splitting
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x4) and B2 (1x4)
    # C = A1 B1 + A2 B2
    
    # A1 B1 is two 2x2 multiplications (Rank 7 each = 14)
    # A2 B2 is 2x1 * 1x4 (Rank 8)
    # Total Rank 22
    
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    # Helper to add to U, V, W
    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val
        
    # Strassen for 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    def add_strassen_block(a_rows, b_rows, c_cols, offset):
        # a_rows: list of (i, j) for A
        # b_rows: list of (j, k) for B
        # c_cols: list of (k, i) for C
        # offset: starting column index in U, V, W
        
        # Map local 2x2 indices to global
        # Local A: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        # Local B: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        
        # m1: (a00+a11)(b00+b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 0, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 0, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 0, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 0, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 0, 1) # C00
        add_w(c_cols[1][0], c_cols[1][1], offset + 0, 1) # C11
        
        # m2: (a10+a11)b00
        # Local A indices:
        # 0: (0,0) -> a00
        # 1: (0,1) -> a01
        # 2: (1,0) -> a10
        # 3: (1,1) -> a11
        # So a11 is index 3. a10 is index 2.
        add_u(a_rows[2][0], a_rows[2][1], offset + 1, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 1, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 1, 1) # b00
        add_w(c_cols[2][0], c_cols[2][1], offset + 1, 1) # C10
        add_w(c_cols[1][0], c_cols[1][1], offset + 1, -1) # C11
        
        # m3: a00(b01-b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 2, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 2, 1) # b01
        add_v(b_rows[3][0], b_rows[3][1], offset + 2, -1) # b11
        add_w(c_cols[3][0], c_cols[3][1], offset + 2, 1) # C01
        add_w(c_cols[1][0], c_cols[1][1], offset + 2, 1) # C11
        
        # m4: a11(b10-b00)
        add_u(a_rows[3][0], a_rows[3][1], offset + 3, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 3, 1) # b10
        add_v(b_rows[0][0], b_rows[0][1], offset + 3, -1) # b00
        add_w(c_cols[0][0], c_cols[0][1], offset + 3, 1) # C00
        add_w(c_cols[2][0], c_cols[2][1], offset + 3, 1) # C10
        
        # m5: (a00+a01)b11
        add_u(a_rows[0][0], a_rows[0][1], offset + 4, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 4, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 4, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 4, -1) # C00
        add_w(c_cols[3][0], c_cols[3][1], offset + 4, 1) # C01
        
        # m6: (a10-a00)(b00+b10)
        add_u(a_rows[2][0], a_rows[2][1], offset + 5, 1)
        add_u(a_rows[0][0], a_rows[0][1], offset + 5, -1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 5, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 5, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 5, 1) # C11
        
        # m7: (a01-a11)(b10+b11)
        add_u(a_rows[1][0], a_rows[1][1], offset + 6, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 6, -1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 6, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 6, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 6, 1) # C00

    # Block 1: A1 (rows 0,1; cols 0,1) * B1a (rows 0,1; cols 0,1)
    a_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    b_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    # C indices: k*n+i
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    # My c_cols list in add_strassen_block expects:
    # c_cols[0] -> C00
    # c_cols[1] -> C11
    # c_cols[2] -> C10
    # c_cols[3] -> C01
    c_cols_1 = [(0,0), (1,1), (0,1), (1,0)] # (k,i)
    add_strassen_block(a_rows_1, b_rows_1, c_cols_1, 0)
    
    # Block 2: A1 * B1b (rows 0,1; cols 2,3)
    b_rows_2 = [(0,2), (0,3), (1,2), (1,3)]
    # C02: k=2, i=0 -> 4
    # C03: k=3, i=0 -> 6
    # C12: k=2, i=1 -> 5
    # C13: k=3, i=1 -> 7
    # Expected order: C02 (k=2,i=0), C13 (k=3,i=1), C12 (k=2,i=1), C03 (k=3,i=0)
    c_cols_2 = [(2,0), (3,1), (2,1), (3,0)]
    add_strassen_block(a_rows_1, b_rows_2, c_cols_2, 7)
    
    # Block 3: A2 (rows 0,1; col 2) * B2 (row 2; cols 0,1,2,3)
    # Rank 8
    # A2: a02 (0,2), a12 (1,2)
    # B2: b20 (2,0), b21 (2,1), b22 (2,2), b23 (2,3)
    # C: all 8
    
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    # Verify U22, V22, W22
    # T is the tensor
    T = np.zeros((p, n, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k, i, j] = 1
                
    # Check: sum_r U[i,j,r] V[j,k,r] W[k,i,r] == T[k, i, j]
    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0
                for r in range(rank_22):
                    total += U22[i*m+j, r] * V22[j*p+k, r] * W22[k*n+i, r]
                if not np.isclose(total, T[k, i, j]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
