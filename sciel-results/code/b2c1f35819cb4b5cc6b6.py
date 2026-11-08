import time
import numpy as np

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    DEADLINE = 130.0

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_list, v_list, w_list):
        U_terms.append(u_list)
        V_terms.append(v_list)
        W_terms.append(w_list)

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    # Helper for Strassen 2x2x2
    # A: rows 0,1 cols 0,1 (relative to block)
    # B: rows 0,1 cols 0,1 (relative to block)
    # C: rows 0,1 cols 0,1 (relative to block)
    def strassen_block(a_row_off, b_col_off, c_row_off, c_col_off):
        # A indices: (a_row_off, 0), (a_row_off+1, 0), (a_row_off, 1), (a_row_off+1, 1)
        # B indices: (0, b_col_off), (0, b_col_off+1), (1, b_col_off), (1, b_col_off+1)
        # C indices: (c_row_off, c_col_off), (c_row_off, c_col_off+1), (c_row_off+1, c_col_off), (c_row_off+1, c_col_off+1)
        
        a00 = (u_idx(a_row_off, 0), 1)
        a01 = (u_idx(a_row_off, 1), 1)
        a10 = (u_idx(a_row_off + 1, 0), 1)
        a11 = (u_idx(a_row_off + 1, 1), 1)
        
        b00 = (v_idx(0, b_col_off), 1)
        b01 = (v_idx(0, b_col_off + 1), 1)
        b10 = (v_idx(1, b_col_off), 1)
        b11 = (v_idx(1, b_col_off + 1), 1)
        
        c00 = (w_idx(c_row_off, c_col_off), 1)
        c01 = (w_idx(c_row_off, c_col_off + 1), 1)
        c10 = (w_idx(c_row_off + 1, c_col_off), 1)
        c11 = (w_idx(c_row_off + 1, c_col_off + 1), 1)
        
        # m1 = (a00+a11)(b00+b11)
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # m2 = (a10+a11)b00
        add_term([a10, a11], [b00], [c10, (w_idx(c_row_off+1, c_col_off+1), -1)])
        
        # m3 = a00(b01-b11)
        add_term([a00], [b01, (v_idx(1, b_col_off+1), -1)], [c01, (w_idx(c_row_off+1, c_col_off+1), 1)])
        
        # m4 = a11(b10-b00)
        add_term([a11], [b10, (v_idx(0, b_col_off), -1)], [c00, c10])
        
        # m5 = (a00+a01)b11
        add_term([a00, a01], [b11], [(w_idx(c_row_off, c_col_off), -1), c01])
        
        # m6 = (a10-a00)(b00+b01)
        add_term([a10, (u_idx(a_row_off, 0), -1)], [b00, b01], [c11])
        
        # m7 = (a01-a11)(b10+b11)
        add_term([a01, (u_idx(a_row_off+1, 1), -1)], [b10, b11], [c00])

    # 1. Handle j=2 part (Rank 1)
    # C[i,k] += A[i,2] * B[2,k]
    add_term([(u_idx(i, 2), 1) for i in range(n)], 
             [(v_idx(2, k), 1) for k in range(p)], 
             [(w_idx(i, k), 1) for i in range(n) for k in range(p)])

    # 2. Handle j in {0,1} part
    # Split k into pairs
    for k0 in range(0, p, 2):
        # Block B: rows 0,1, cols k0, k0+1
        
        # Part 2a: A rows 0,1 (2x2) * B (2x2) -> C rows 0,1 (2x2)
        strassen_block(0, k0, 0, k0)
        
        # Part 2b: A row 2 (1x2) * B (2x2) -> C row 2 (1x2)
        # Use 4 rank-1 terms
        # A[2,0]*B[0,k0] -> C[2,k0]
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0), 1)], [(w_idx(2, k0), 1)])
        # A[2,1]*B[1,k0] -> C[2,k0]
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0), 1)], [(w_idx(2, k0), 1)])
        
        # A[2,0]*B[0,k0+1] -> C[2,k0+1]
        add_term([(u_idx(2, 0), 1)], [(v_idx(0, k0 + 1), 1)], [(w_idx(2, k0 + 1), 1)])
        # A[2,1]*B[1,k0+1] -> C[2,k0+1]
        add_term([(u_idx(2, 1), 1)], [(v_idx(1, k0 + 1), 1)], [(w_idx(2, k0 + 1), 1)])

    rank = len(W_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx in range(rank):
        for (flat_idx, coeff) in U_terms[idx]:
            U[flat_idx][idx] = coeff
        for (flat_idx, coeff) in V_terms[idx]:
            V[flat_idx][idx] = coeff
        for (flat_idx, coeff) in W_terms[idx]:
            W[flat_idx][idx] = coeff

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    def verify_decomp(Ua, Va, Wa):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
