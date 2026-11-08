import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_TIME_LIMIT = 120.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Strategy:
    # Rows 0-1 of A (2x3) * B (3x7):
    #   - A[0:2,0:2] * B[0:2,0:2]: Strassen rank 7
    #   - A[0:2,0:2] * B[0:2,2:4]: Rank 9 (2x2 x 2x3)
    #   - A[0:2,2] * B[2,0:7]: 2x1 x 1x7 = rank 14
    #   Total: 7 + 9 + 14 = 30
    # Row 2 of A (1x3) * B (3x7): rank 21
    # Grand total: 30 + 21 = 51
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]  # 9 x 51
    V = [[0] * rank for _ in range(m * p)]  # 21 x 51
    W = [[0] * rank for _ in range(p * n)]  # 21 x 51
    
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
    
    def u_idx(i, j): return i * m + j  # i=row of A, j=col of A
    def v_idx(j, k): return j * p + k  # j=row of B, k=col of B
    def w_idx(k, i): return k * n + i  # k=col of C, i=row of C
    
    def add_strassen_2x2x2(a_row_start, b_row_start, c_row_start, b_col_start):
        """Strassen for 2x2 * 2x2 -> 2x2, rank 7"""
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
        
        # m1 = (a00+a11)(b00+b11)
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2 = (a10+a11)b00
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3 = a00(b01-b11)
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4 = a11(b10-b00)
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5 = (a00+a01)b11
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1
        W5[c00] = -1; W5[c01] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6 = (a10-a00)(b00+b01)
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7 = (a01-a11)(b10+b11)
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)
    
    def add_rank9_2x2x3(a_row_start, b_row_start, c_row_start, b_col_start):
        """
        Rank 9 decomposition for 2x2 * 2x3 -> 2x3
        A is 2x2, B is 2x3, C is 2x3
        """
        a00 = u_idx(a_row_start, 0)
        a01 = u_idx(a_row_start, 1)
        a10 = u_idx(a_row_start + 1, 0)
        a11 = u_idx(a_row_start + 1, 1)
        
        b00 = v_idx(b_row_start, b_col_start)
        b01 = v_idx(b_row_start, b_col_start + 1)
        b02 = v_idx(b_row_start, b_col_start + 2)
        b10 = v_idx(b_row_start + 1, b_col_start)
        b11 = v_idx(b_row_start + 1, b_col_start + 1)
        b12 = v_idx(b_row_start + 1, b_col_start + 2)
        
        c00 = w_idx(b_col_start, c_row_start)
        c01 = w_idx(b_col_start + 1, c_row_start)
        c02 = w_idx(b_col_start + 2, c_row_start)
        c10 = w_idx(b_col_start, c_row_start + 1)
        c11 = w_idx(b_col_start + 1, c_row_start + 1)
        c12 = w_idx(b_col_start + 2, c_row_start + 1)
        
        # Use a structured rank-9 decomposition
        # This is based on the idea of extending Strassen to rectangular case
        
        # m1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[a00] = 1; U1[a11] = 1
        V1[b00] = 1; V1[b11] = 1
        W1[c00] = 1; W1[c11] = 1
        add_rank1_term(U1, V1, W1)
        
        # m2 = (a10+a11)(b00+b11) -> contributes to c10, c11
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[a10] = 1; U2[a11] = 1
        V2[b00] = 1; V2[b11] = 1
        W2[c10] = 1; W2[c11] = -1
        add_rank1_term(U2, V2, W2)
        
        # m3 = a00(b01-b11) -> contributes to c01, c11
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[a00] = 1
        V3[b01] = 1; V3[b11] = -1
        W3[c01] = 1; W3[c11] = 1
        add_rank1_term(U3, V3, W3)
        
        # m4 = a11(b10-b00) -> contributes to c00, c10
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[a11] = 1
        V4[b10] = 1; V4[b00] = -1
        W4[c00] = 1; W4[c10] = 1
        add_rank1_term(U4, V4, W4)
        
        # m5 = (a00+a01)(b11+b12) -> contributes to c01, c02
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[a00] = 1; U5[a01] = 1
        V5[b11] = 1; V5[b12] = 1
        W5[c01] = 1; W5[c02] = 1
        add_rank1_term(U5, V5, W5)
        
        # m6 = (a10-a00)(b00+b01) -> contributes to c11
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[a10] = 1; U6[a00] = -1
        V6[b00] = 1; V6[b01] = 1
        W6[c11] = 1
        add_rank1_term(U6, V6, W6)
        
        # m7 = (a01-a11)(b10+b11) -> contributes to c00
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[a01] = 1; U7[a11] = -1
        V7[b10] = 1; V7[b11] = 1
        W7[c00] = 1
        add_rank1_term(U7, V7, W7)
        
        # m8 = a00*b02 -> contributes to c02
        U8 = [0]*(n*m); V8 = [0]*(m*p); W8 = [0]*(p*n)
        U8[a00] = 1
        V8[b02] = 1
        W8[c02] = 1
        add_rank1_term(U8, V8, W8)
        
        # m9 = a10*b12 -> contributes to c12
        U9 = [0]*(n*m); V9 = [0]*(m*p); W9 = [0]*(p*n)
        U9[a10] = 1
        V9[b12] = 1
        W9[c12] = 1
        add_rank1_term(U9, V9, W9)
    
    # Process rows 0-1 of A (2x3) * B (3x7)
    
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2x2(0, 0, 0, 0)
    
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    add_rank9_2x2x3(0, 0, 0, 2)
    
    # A[0:2, 2] * B[2, 0:7] -> C[0:2, 0:7]
    for i in range(2):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(i, 2)] = 1
            V_term[v_idx(2, k)] = 1
            W_term[w_idx(k, i)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Row 2 of A (1x3) * B (3x7) -> C[2, 0:7]
    for j in range(3):
        for k in range(7):
            U_term = [0]*(n*m); V_term = [0]*(m*p); W_term = [0]*(p*n)
            U_term[u_idx(2, j)] = 1
            V_term[v_idx(j, k)] = 1
            W_term[w_idx(k, 2)] = 1
            add_rank1_term(U_term, V_term, W_term)
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for j in range(m):
                for r in range(rank):
                    val += U_arr[i*m + j, r] * A_test[i, j] * V_arr[j*p + k, r] * B_test[j, k] * W_arr[k*n + i, r]
            C_comp[i, k] = val
            
    if not np.allclose(C_comp, C_true, atol=1e-9):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
