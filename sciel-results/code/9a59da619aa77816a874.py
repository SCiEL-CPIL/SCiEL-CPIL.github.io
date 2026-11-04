"""Matrix multiplication 2x3x4 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_u(i, j, c, val):
        U[i * m + j, c] += val
    def add_v(j, k, c, val):
        V[j * p + k, c] += val
    def add_w(k, i, c, val):
        W[k * n + i, c] += val
        
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        a_rows, a_cols: indices for A (i, j)
        b_rows, b_cols: indices for B (j, k)
        c_rows, c_cols: indices for C (i, k)
        
        W index convention: W[k*n + i, r]
        """
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        # c_rows correspond to i, c_cols correspond to k
        c00_i, c00_k = c_rows[0], c_cols[0]
        c01_i, c01_k = c_rows[0], c_cols[1]
        c10_i, c10_k = c_rows[1], c_cols[0]
        c11_i, c11_k = c_rows[1], c_cols[1]
        
        # M1 = (a00+a11)(b00+b11)
        add_u(a00[0], a00[1], offset+0, 1)
        add_u(a11[0], a11[1], offset+0, 1)
        add_v(b00[0], b00[1], offset+0, 1)
        add_v(b11[0], b11[1], offset+0, 1)
        # C00 += M1, C11 += M1
        add_w(c00_k, c00_i, offset+0, 1)
        add_w(c11_k, c11_i, offset+0, 1)
        
        # M2 = (a10+a11)(b00)
        add_u(a10[0], a10[1], offset+1, 1)
        add_u(a11[0], a11[1], offset+1, 1)
        add_v(b00[0], b00[1], offset+1, 1)
        # C10 += M2, C11 -= M2
        add_w(c10_k, c10_i, offset+1, 1)
        add_w(c11_k, c11_i, offset+1, -1)
        
        # M3 = a00(b01-b11)
        add_u(a00[0], a00[1], offset+2, 1)
        add_v(b01[0], b01[1], offset+2, 1)
        add_v(b11[0], b11[1], offset+2, -1)
        # C01 += M3, C11 += M3
        add_w(c01_k, c01_i, offset+2, 1)
        add_w(c11_k, c11_i, offset+2, 1)
        
        # M4 = a11(b10-b00)
        add_u(a11[0], a11[1], offset+3, 1)
        add_v(b10[0], b10[1], offset+3, 1)
        add_v(b00[0], b00[1], offset+3, -1)
        # C00 -= M4, C10 += M4
        add_w(c00_k, c00_i, offset+3, -1)
        add_w(c10_k, c10_i, offset+3, 1)
        
        # M5 = (a00+a01)(b11)
        add_u(a00[0], a00[1], offset+4, 1)
        add_u(a01[0], a01[1], offset+4, 1)
        add_v(b11[0], b11[1], offset+4, 1)
        # C00 -= M5, C01 += M5
        add_w(c00_k, c00_i, offset+4, -1)
        add_w(c01_k, c01_i, offset+4, 1)
        
        # M6 = (a10-a00)(b00+b10)
        add_u(a10[0], a10[1], offset+5, 1)
        add_u(a00[0], a00[1], offset+5, -1)
        add_v(b00[0], b00[1], offset+5, 1)
        add_v(b10[0], b10[1], offset+5, 1)
        # C11 += M6
        add_w(c11_k, c11_i, offset+5, 1)
        
        # M7 = (a01-a11)(b10+b11)
        add_u(a01[0], a01[1], offset+6, 1)
        add_u(a11[0], a11[1], offset+6, -1)
        add_v(b10[0], b10[1], offset+6, 1)
        add_v(b11[0], b11[1], offset+6, 1)
        # C00 += M7
        add_w(c00_k, c00_i, offset+6, 1)
        
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 7)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # 2x1 * 1x4: 8 independent products
    offset = 14
    for i in range(2):
        for k in range(4):
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    def verify(U_mat, V_mat, W_mat):
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 24
        U_school = np.zeros((n * m, 24))
        V_school = np.zeros((m * p, 24))
        W_school = np.zeros((p * n, 24))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, idx] = 1
                    V_school[j * p + k, idx] = 1
                    W_school[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
