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
    
    # Rank-22 decomposition using Strassen blocks + outer product
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7)
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] (Rank 8)
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
        
    def add_strassen_block(a_i_list, a_j_list, b_j_list, b_k_list, c_i_list, c_k_list, offset):
        """
        Adds a Strassen decomposition for a 2x2 * 2x2 block.
        
        Parameters:
        a_i_list: [i0, i1] - row indices for A
        a_j_list: [j0, j1] - col indices for A
        b_j_list: [j0, j1] - row indices for B
        b_k_list: [k0, k1] - col indices for B
        c_i_list: [i0, i1] - row indices for C (i dimension)
        c_k_list: [k0, k1] - col indices for C (k dimension)
        offset: starting column index
        
        Strassen formulas for C = A*B where:
        A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)(b00)
        M3 = a00(b01-b11)
        M4 = a11(b10-b00)
        M5 = (a00+a01)(b11)
        M6 = (a10-a00)(b00+b10)
        M7 = (a01-a11)(b10+b11)
        
        C00 = M1 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        i0, i1 = c_i_list
        k0, k1 = c_k_list
        j0, j1 = a_j_list  # same as b_j_list for this problem
        
        # M1 = (a00+a11)(b00+b11)
        # A: a00 = (i0, j0), a11 = (i1, j1)
        # B: b00 = (j0, k0), b11 = (j1, k1)
        add_u(i0, j0, offset+0, 1)
        add_u(i1, j1, offset+0, 1)
        add_v(j0, k0, offset+0, 1)
        add_v(j1, k1, offset+0, 1)
        # C00: (i0, k0), C11: (i1, k1)
        add_w(k0, i0, offset+0, 1)
        add_w(k1, i1, offset+0, 1)
        
        # M2 = (a10+a11)(b00)
        # A: a10 = (i1, j0), a11 = (i1, j1)
        # B: b00 = (j0, k0)
        add_u(i1, j0, offset+1, 1)
        add_u(i1, j1, offset+1, 1)
        add_v(j0, k0, offset+1, 1)
        # C10: (i1, k0), C11: (i1, k1)
        add_w(k0, i1, offset+1, 1)
        add_w(k1, i1, offset+1, -1)
        
        # M3 = a00(b01-b11)
        # A: a00 = (i0, j0)
        # B: b01 = (j0, k1), b11 = (j1, k1)
        add_u(i0, j0, offset+2, 1)
        add_v(j0, k1, offset+2, 1)
        add_v(j1, k1, offset+2, -1)
        # C01: (i0, k1), C11: (i1, k1)
        add_w(k1, i0, offset+2, 1)
        add_w(k1, i1, offset+2, 1)
        
        # M4 = a11(b10-b00)
        # A: a11 = (i1, j1)
        # B: b10 = (j1, k0), b00 = (j0, k0)
        add_u(i1, j1, offset+3, 1)
        add_v(j1, k0, offset+3, 1)
        add_v(j0, k0, offset+3, -1)
        # C00: (i0, k0), C10: (i1, k0)
        add_w(k0, i0, offset+3, -1)
        add_w(k0, i1, offset+3, 1)
        
        # M5 = (a00+a01)(b11)
        # A: a00 = (i0, j0), a01 = (i0, j1)
        # B: b11 = (j1, k1)
        add_u(i0, j0, offset+4, 1)
        add_u(i0, j1, offset+4, 1)
        add_v(j1, k1, offset+4, 1)
        # C00: (i0, k0), C01: (i0, k1)
        add_w(k0, i0, offset+4, -1)
        add_w(k1, i0, offset+4, 1)
        
        # M6 = (a10-a00)(b00+b10)
        # A: a10 = (i1, j0), a00 = (i0, j0)
        # B: b00 = (j0, k0), b10 = (j1, k0)
        add_u(i1, j0, offset+5, 1)
        add_u(i0, j0, offset+5, -1)
        add_v(j0, k0, offset+5, 1)
        add_v(j1, k0, offset+5, 1)
        # C11: (i1, k1)
        add_w(k1, i1, offset+5, 1)
        
        # M7 = (a01-a11)(b10+b11)
        # A: a01 = (i0, j1), a11 = (i1, j1)
        # B: b10 = (j1, k0), b11 = (j1, k1)
        add_u(i0, j1, offset+6, 1)
        add_u(i1, j1, offset+6, -1)
        add_v(j1, k0, offset+6, 1)
        add_v(j1, k1, offset+6, 1)
        # C00: (i0, k0)
        add_w(k0, i0, offset+6, 1)
    
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
            
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False, (i, j, k, total)
        return True, None

    valid, err = verify(U, V, W)
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 24
        rank_fb = n * m * p
        U_school = np.zeros((n * m, rank_fb))
        V_school = np.zeros((m * p, rank_fb))
        W_school = np.zeros((p * n, rank_fb))
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
