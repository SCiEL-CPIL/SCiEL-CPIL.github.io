# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
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
        
    # Strassen 2x2 multiplication helper
    # a_idx: [(i0,j0), (i0,j1), (i1,j0), (i1,j1)]
    # b_idx: [(j0,k0), (j0,k1), (j1,k0), (j1,k1)]
    # c_idx: [(k0,i0), (k1,i0), (k0,i1), (k1,i1)]
    # Note: c_idx order is (C00, C01, C10, C11) where C_ij is row i, col j
    # But W index is k*n+i. So C00 is (k0, i0), C01 is (k1, i0), C10 is (k0, i1), C11 is (k1, i1)
    
    def add_strassen(a_idx, b_idx, c_idx, offset):
        # m1 = (a00+a11)(b00+b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+0, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+0, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+0, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+0, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+0, 1)   # C00
        add_w(c_idx[3][0], c_idx[3][1], offset+0, 1)   # C11
        
        # m2 = (a10+a11)(b00)
        add_u(a_idx[2][0], a_idx[2][1], offset+1, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+1, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+1, 1)
        add_w(c_idx[2][0], c_idx[2][1], offset+1, 1)   # C10
        add_w(c_idx[3][0], c_idx[3][1], offset+1, -1)  # C11
        
        # m3 = a00(b01-b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+2, 1)
        add_v(b_idx[1][0], b_idx[1][1], offset+2, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+2, -1)
        add_w(c_idx[1][0], c_idx[1][1], offset+2, 1)   # C01
        add_w(c_idx[3][0], c_idx[3][1], offset+2, 1)   # C11
        
        # m4 = a11(b10-b00)
        add_u(a_idx[3][0], a_idx[3][1], offset+3, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+3, 1)
        add_v(b_idx[0][0], b_idx[0][1], offset+3, -1)
        add_w(c_idx[0][0], c_idx[0][1], offset+3, 1)   # C00
        add_w(c_idx[2][0], c_idx[2][1], offset+3, 1)   # C10
        
        # m5 = (a00+a01)(b11)
        add_u(a_idx[0][0], a_idx[0][1], offset+4, 1)
        add_u(a_idx[1][0], a_idx[1][1], offset+4, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+4, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+4, -1)  # C00
        add_w(c_idx[1][0], c_idx[1][1], offset+4, 1)   # C01
        
        # m6 = (a10-a00)(b00+b10)
        add_u(a_idx[2][0], a_idx[2][1], offset+5, 1)
        add_u(a_idx[0][0], a_idx[0][1], offset+5, -1)
        add_v(b_idx[0][0], b_idx[0][1], offset+5, 1)
        add_v(b_idx[2][0], b_idx[2][1], offset+5, 1)
        add_w(c_idx[3][0], c_idx[3][1], offset+5, 1)   # C11
        
        # m7 = (a01-a11)(b10+b11)
        add_u(a_idx[1][0], a_idx[1][1], offset+6, 1)
        add_u(a_idx[3][0], a_idx[3][1], offset+6, -1)
        add_v(b_idx[2][0], b_idx[2][1], offset+6, 1)
        add_v(b_idx[3][0], b_idx[3][1], offset+6, 1)
        add_w(c_idx[0][0], c_idx[0][1], offset+6, 1)   # C00

    # Block 1: A1 (cols 0,1) * B1a (rows 0,1, cols 0,1)
    a_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    b_idx1 = [(0,0), (0,1), (1,0), (1,1)]
    c_idx1 = [(0,0), (1,0), (0,1), (1,1)]
    add_strassen(a_idx1, b_idx1, c_idx1, 0)
    
    # Block 2: A1 (cols 0,1) * B1b (rows 0,1, cols 2,3)
    b_idx2 = [(0,2), (0,3), (1,2), (1,3)]
    c_idx2 = [(2,0), (3,0), (2,1), (3,1)]
    add_strassen(a_idx1, b_idx2, c_idx2, 7)
    
    # Block 3: A2 (col 2) * B2 (row 2, cols 0-3)
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    # Verification
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
