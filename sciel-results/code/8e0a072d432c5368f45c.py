import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1_term(U_vec, V_vec, W_vec):
        nonlocal col
        if col >= rank:
            return False
        for i, val in enumerate(U_vec):
            if val != 0:
                U[i][col] += val
        for i, val in enumerate(V_vec):
            if val != 0:
                V[i][col] += val
        for i, val in enumerate(W_vec):
            if val != 0:
                W[i][col] += val
        col += 1
        return True
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    def add_strassen_2x2x2(a_row_start, b_row_start, c_row_start, b_col_start):
        a00 = u_idx(a_row_start, 0)
        a01 = u_idx(a_row_start, 1)
        a10 = u_idx(a_row_start + 1, 0)
        a11 = u_idx(a_row_start + 1, 1)
        
        b00 = v_idx(b_row_start, b_col_start)
        b01 = v_idx(b_row_start, b_col_start + 1)
        b10 = v_idx(b_row_start + 1, b_col_start)
        b11 = v_idx(b_row_start + 1, b_col_start + 1)
        
        c00 = w_idx(b_col_start, c_row_start)
        c01 = w_idx(b_col_start + 1, c_row_start)
        c10 = w_idx(b_col_start, c_row_start + 1)
        c11 = w_idx(b_col_start + 1, c_row_start + 1)
        
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)

    def add_schoolbook_2x2x1(a_row_start, b_row_start, c_row_start, b_col):
        for i in range(2):
            for j in range(2):
                U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
                U_term[u_idx(a_row_start + i, j)] = 1
                V_term[v_idx(b_row_start + j, b_col)] = 1
                W_term[w_idx(b_col, c_row_start + i)] = 1
                add_rank1_term(U_term, V_term, W_term)

    add_strassen_2x2x2(0, 0, 0, 0)
    add_strassen_2x2x2(0, 0, 0, 2)
    add_strassen_2x2x2(0, 0, 0, 4)
    add_schoolbook_2x2x1(0, 0, 0, 6)
    
    for i in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(k, i)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    for j in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(k, 2)] = 1
            add_rank1_term(U_term, V_term, W_term)
            
    return {"U": U, "V": V, "W": W}
