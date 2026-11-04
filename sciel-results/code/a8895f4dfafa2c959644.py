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
    
    # Strassen-based rank-22 decomposition
    # Split: A = [A1 | A2], A1 is 2x2, A2 is 2x1
    #         B = [B1; B2], B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    # A1*B1: two 2x2 multiplications, rank 7 each = 14
    # A2*B2: 2x1 * 1x4, rank 8
    # Total: 22
    
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val
    
    # Strassen 2x2 multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)(b11)
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 - m5 + m7 + m4
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # a_idx: local A indices [a00, a01, a10, a11] -> (i,j) pairs
    # b_idx: local B indices [b00, b01, b10, b11] -> (j,k) pairs
    # c_idx: local C indices [C00, C01, C10, C11] -> (k,i) pairs
    
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # m1: (a00+a11)(b00+b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+0, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+0, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+0, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+0, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+0, 1)   # C00
        add_w(c_idx[3][0], c_idx[3][1], offset+0, 1)   # C11
        
        # m2: (a10+a11)(b00)
        add_u(a_idx[2][0], a_idx[2][1], offset+1, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+1, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+1, 1)
        add_w(c_idx[2][0], c_idx[2][1], offset+1, 1)   # C10
        add_w(c_idx[3][0], c_idx[3][1], offset+1, -1)  # C11
        
        # m3: a00(b01-b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+2, 1)
        add_v(b_idx[1][0], b_idx[1][1], offset+2, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+2, -1)
        add_w(c_idx[1][0], c_idx[1][1], offset+2, 1)   # C01
        add_w(c_idx[3][0], c_idx[3][1], offset+2, 1)   # C11
        
        # m4: a11(b10-b00)
        add_u(a_idx[3][0], a_idx[3][1], offset+3, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+3, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+3, -1)
        add_w(c_idx[0][0], c_idx[0][1], offset+3, 1)   # C00
        add_w(c_idx[2][0], c_idx[2][1], offset+3, 1)   # C10
        
        # m5: (a00+a01)(b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+4, 1)
        add_u(a_idx[1][0], a_idx[1][1], offset+4, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+4, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+4, -1)  # C00
        add_w(c_idx[1][0], c_idx[1][1], offset+4, 1)   # C01
        
        # m6: (a10-a00)(b00+b10)
        add_u(a_idx[2][0], a_idx[2][1], offset+5, 1)
        add_u(a_idx[0][0], a_idx[0][1], offset+5, -1)
        add_v(b_idx[0][0], b_idx[0][1], offset+5, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+5, 1)
        add_w(c_idx[3][0], c_idx[3][1], offset+5, 1)   # C11
        
        # m7: (a01-a11)(b10+b11)
        add_u(a_idx[1][0], a_idx[1][1], offset+6, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+6, -1)
        add_v(b_idx[2][0], b_idx[2][1], offset+6, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+6, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+6, 1)   # C00
    
    # Block 1: A1 (cols 0,1) * B1a (rows 0,1, cols 0,1)
    # A1: a00=(0,0), a01=(0,1), a10=(1,0), a11=(1,1)
    a_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    # B1a: b00=(0,0), b01=(0,1), b10=(1,0), b11=(1,1)
    b_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    # C: C00=(k=0,i=0), C01=(k=1,i=0), C10=(k=0,i=1), C11=(k=1,i=1)
    c_idx1 = [(0,0), (1,0), (0,1), (1,1)]
    add_strassen(a_idx1, b_idx1, c_idx1, 0)
    
    # Block 2: A1 (cols 0,1) * B1b (rows 0,1, cols 2,3)
    b_idx2 = [(0,2), (0,3), (1,2), (1,3)]
    # C: C02=(k=2,i=0), C03=(k=3,i=0), C12=(k=2,i=1), C13=(k=3,i=1)
    c_idx2 = [(2,0), (3,0), (2,1), (3,1)]
    add_strassen(a_idx1, b_idx2, c_idx2, 7)
    
    # Block 3: A2 (col 2) * B2 (row 2, cols 0-3)
    # A2: a02=(0,2), a12=(1,2)
    # B2: b20=(2,0), b21=(2,1), b22=(2,2), b23=(2,3)
    # Each output C[i,k] = a_i2 * b_2k, rank 8
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
    
    # Verify
    T = np.ones((p, n, m))
    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0.0
                for r in range(rank_22):
                    total += U22[i*m+j, r] * V22[j*p+k, r] * W22[k*n+i, r]
                if not np.isclose(total, T[k, i, j], atol=1e-9):
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
