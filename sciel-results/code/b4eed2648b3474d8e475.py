import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    R = 28
    
    U = np.zeros((nm, R), dtype=int)
    V = np.zeros((mp, R), dtype=int)
    W = np.zeros((pn, R), dtype=int)
    
    def add_strassen_block(a_rows, b_cols, c_block):
        # a_rows: (i1, i2) indices of rows in A for the 2x2 block
        # b_cols: (k1, k2) indices of cols in C (which are rows in B block)
        # Actually, let's pass the specific indices of A, B, C involved.
        # A_block: 2x2. Rows i1, i2. Cols j1, j2.
        # B_block: 2x2. Rows j1, j2. Cols k1, k2.
        # C_block: 2x2. Rows i1, i2. Cols k1, k2.
        pass

    # Helper to get linear index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i  # W[k*n+i]

    col = 0
    
    # Strassen terms definition for X (2x2) * Y (2x2) -> Z (2x2)
    # X indices: 00, 01, 10, 11
    # Y indices: 00, 01, 10, 11
    # Z indices: 00, 01, 10, 11
    
    def apply_strassen(a_i, a_j, b_j, b_k, c_i, c_k):
        nonlocal col
        # a_i: [i1, i2], a_j: [j1, j2]
        # b_j: [j1, j2], b_k: [k1, k2]
        # c_i: [i1, i2], c_k: [k1, k2]
        
        # Mapping of logical (0,0), (0,1), (1,0), (1,1) to physical indices
        a_00 = u_idx(a_i[0], a_j[0])
        a_01 = u_idx(a_i[0], a_j[1])
        a_10 = u_idx(a_i[1], a_j[0])
        a_11 = u_idx(a_i[1], a_j[1])
        
        b_00 = v_idx(b_j[0], b_k[0])
        b_01 = v_idx(b_j[0], b_k[1])
        b_10 = v_idx(b_j[1], b_k[0])
        b_11 = v_idx(b_j[1], b_k[1])
        
        c_00 = w_idx(c_k[0], c_i[0])
        c_01 = w_idx(c_k[1], c_i[0])
        c_10 = w_idx(c_k[0], c_i[1])
        c_11 = w_idx(c_k[1], c_i[1])
        
        # M1 = (x00 + x11)(y00 + y11)
        U[a_00, col] = 1; U[a_11, col] = 1
        V[b_00, col] = 1; V[b_11, col] = 1
        W[c_00, col] = 1; W[c_11, col] = 1
        col += 1
        
        # M2 = (x10 + x11)y00
        U[a_10, col] = 1; U[a_11, col] = 1
        V[b_00, col] = 1
        W[c_10, col] = 1
        col += 1
        
        # M3 = x00(y01 - y11)
        U[a_00, col] = 1
        V[b_01, col] = 1; V[b_11, col] = -1
        W[c_01, col] = 1
        col += 1
        
        # M4 = x11(y10 - y00)
        U[a_11, col] = 1
        V[b_10, col] = 1; V[b_00, col] = -1
        W[c_00, col] = 1; W[c_10, col] = 1
        col += 1
        
        # M5 = (x00 + x01)y11
        U[a_00, col] = 1; U[a_01, col] = 1
        V[b_11, col] = 1
        W[c_00, col] = -1; W[c_01, col] = 1
        col += 1
        
        # M6 = (x10 - x00)(y00 + y01)
        U[a_10, col] = 1; U[a_00, col] = -1
        V[b_00, col] = 1; V[b_01, col] = 1
        W[c_11, col] = 1
        col += 1
        
        # M7 = (x01 - x11)(y10 + y11)
        U[a_01, col] = 1; U[a_11, col] = -1
        V[b_10, col] = 1; V[b_11, col] = 1
        W[c_00, col] = 1; W[c_01, col] = -1
        col += 1

    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # A rows 0,1; Cols 0,1. B Rows 0,1; Cols 0,1. C Rows 0,1; Cols 0,1.
    apply_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    # A rows 0,1; Cols 2,3. B Rows 2,3; Cols 0,1. C Rows 0,1; Cols 0,1.
    apply_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # 3. A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    # A rows 0,1; Cols 0,1. B Rows 0,1; Cols 2,3. C Rows 0,1; Cols 2,3.
    apply_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # 4. A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    # A rows 0,1; Cols 2,3. B Rows 2,3; Cols 2,3. C Rows 0,1; Cols 2,3.
    apply_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
