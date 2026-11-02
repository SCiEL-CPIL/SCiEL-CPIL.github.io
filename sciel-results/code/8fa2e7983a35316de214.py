import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def get_A_idx(i, j):
        return i * m + j
        
    def get_B_idx(j, k):
        return j * p + k
        
    def get_C_idx(k, i):
        return k * n + i

    def strassen_block(i_off, j_off, k_off):
        # i_off: 0
        # j_off: 0 or 2
        # k_off: 0, 2, 4
        # Local indices 0,1 for i, j, k in the block
        
        # M1 = (A00+A11)(B00+B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+0, k_off+0)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        # C00 gets +1, C11 gets +1
        w[get_C_idx(k_off+0, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+0, k_off+0)] = 1
        # C10 gets +1, C11 gets -1
        w[get_C_idx(k_off+0, i_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = -1
        add_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = -1
        # C01 gets +1, C11 gets +1
        w[get_C_idx(k_off+1, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+0)] = -1
        # C10 gets +1, C11 gets +1
        w[get_C_idx(k_off+0, i_off+1)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        u[get_A_idx(i_off+0, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        # C01 gets +1, C11 gets -1
        w[get_C_idx(k_off+1, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = -1
        add_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        u[get_A_idx(i_off+0, j_off+0)] = -1
        v[get_B_idx(j_off+0, k_off+0)] = 1
        v[get_B_idx(j_off+0, k_off+1)] = 1
        # C11 gets +1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+1)] = 1
        u[get_A_idx(i_off+1, j_off+1)] = -1
        v[get_B_idx(j_off+1, k_off+0)] = 1
        v[get_B_idx(j_off+1, k_off+1)] = 1
        # C00 gets +1, C11 gets +1
        w[get_C_idx(k_off+0, i_off+0)] = 1
        w[get_C_idx(k_off+1, i_off+1)] = 1
        add_term(u, v, w)

    def schoolbook_221_block(i_off, j_off, k_off):
        # i_off: 0
        # j_off: 0 or 2
        # k_off: 6
        # B is 2x1 at (j_off, k_off)
        # A is 2x2 at (i_off, j_off)
        # C is 2x1 at (k_off, i_off)
        
        # M1 = A00 * B0
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off)] = 1
        w[get_C_idx(k_off, i_off+0)] = 1
        add_term(u, v, w)
        
        # M2 = A01 * B1
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+0, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off)] = 1
        w[get_C_idx(k_off, i_off+0)] = 1
        add_term(u, v, w)
        
        # M3 = A10 * B0
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+0)] = 1
        v[get_B_idx(j_off+0, k_off)] = 1
        w[get_C_idx(k_off, i_off+1)] = 1
        add_term(u, v, w)
        
        # M4 = A11 * B1
        u = [0]*(n*m)
        v = [0]*(m*p)
        w = [0]*(p*n)
        u[get_A_idx(i_off+1, j_off+1)] = 1
        v[get_B_idx(j_off+1, k_off)] = 1
        w[get_C_idx(k_off, i_off+1)] = 1
        add_term(u, v, w)

    # Call blocks
    # Part 1: k in 0..1 (Strassen)
    strassen_block(0, 0, 0)
    strassen_block(0, 2, 0)
    
    # Part 2: k in 2..3 (Strassen)
    strassen_block(0, 0, 2)
    strassen_block(0, 2, 2)
    
    # Part 3: k in 4..5 (Strassen)
    strassen_block(0, 0, 4)
    strassen_block(0, 2, 4)
    
    # Part 4: k in 6 (Schoolbook)
    schoolbook_221_block(0, 0, 6)
    schoolbook_221_block(0, 2, 6)
    
    rank = len(U_cols)
    
    # Construct matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r in range(rank):
        u_vec = U_cols[r]
        v_vec = V_cols[r]
        w_vec = W_cols[r]
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
