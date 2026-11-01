import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] = val

    # --- Block 1: M_2,2,2 (Strassen) for C cols 0,1 from B rows 0,1 cols 0,1 and A cols 0,1 ---
    # Global U rows for A cols 0,1: i*3+j
    # i=0: j=0->0, j=1->1
    # i=1: j=0->3, j=1->4
    u_rows_b1 = [0, 1, 3, 4] # a00, a01, a10, a11
    
    # Global V rows for B rows 0,1 cols 0,1: j*5+k
    # j=0: k=0->0, k=1->1
    # j=1: k=0->5, k=1->6
    v_rows_b1 = [0, 1, 5, 6] # b00, b01, b10, b11
    
    # Global W rows for C cols 0,1: k*2+i
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    w_rows_b1 = [0, 1, 2, 3] # c00, c10, c01, c11
    
    # Strassen Coeffs for M_2,2,2
    # Col 0: M1
    set_val(U, u_rows_b1[0], 0, 1); set_val(U, u_rows_b1[3], 0, 1)
    set_val(V, v_rows_b1[0], 0, 1); set_val(V, v_rows_b1[3], 0, 1)
    set_val(W, w_rows_b1[0], 0, 1); set_val(W, w_rows_b1[3], 0, 1)
    
    # Col 1: M2
    set_val(U, u_rows_b1[2], 1, 1); set_val(U, u_rows_b1[3], 1, 1)
    set_val(V, v_rows_b1[0], 1, 1)
    set_val(W, w_rows_b1[1], 1, 1); set_val(W, w_rows_b1[3], 1, -1)
    
    # Col 2: M3
    set_val(U, u_rows_b1[0], 2, 1)
    set_val(V, v_rows_b1[1], 2, 1); set_val(V, v_rows_b1[3], 2, -1)
    set_val(W, w_rows_b1[2], 2, 1); set_val(W, w_rows_b1[3], 2, 1)
    
    # Col 3: M4
    set_val(U, u_rows_b1[3], 3, 1)
    set_val(V, v_rows_b1[2], 3, 1); set_val(V, v_rows_b1[0], 3, -1)
    set_val(W, w_rows_b1[0], 3, 1); set_val(W, w_rows_b1[1], 3, 1)
    
    # Col 4: M5
    set_val(U, u_rows_b1[0], 4, 1); set_val(U, u_rows_b1[1], 4, 1)
    set_val(V, v_rows_b1[3], 4, 1)
    set_val(W, w_rows_b1[0], 4, -1); set_val(W, w_rows_b1[2], 4, 1)
    
    # Col 5: M6
    set_val(U, u_rows_b1[2], 5, 1); set_val(U, u_rows_b1[0], 5, -1)
    set_val(V, v_rows_b1[0], 5, 1); set_val(V, v_rows_b1[1], 5, 1)
    set_val(W, w_rows_b1[3], 5, 1)
    
    # Col 6: M7
    set_val(U, u_rows_b1[1], 6, 1); set_val(U, u_rows_b1[3], 6, -1)
    set_val(V, v_rows_b1[2], 6, 1); set_val(V, v_rows_b1[3], 6, 1)
    set_val(W, w_rows_b1[0], 6, 1)

    # --- Block 2: M_2,2,2 (Strassen) for C cols 2,3 from B rows 0,1 cols 2,3 and A cols 0,1 ---
    # U rows same as B1
    u_rows_b2 = u_rows_b1
    
    # V rows for B rows 0,1 cols 2,3
    v_rows_b2 = [2, 3, 7, 8] # b02, b03, b12, b13
    
    # W rows for C cols 2,3
    w_rows_b2 = [4, 5, 6, 7] # c20, c30, c21, c31
    
    # Fill cols 7-13 with same pattern as B1
    for c_offset in range(7):
        c_global = 7 + c_offset
        # Copy U
        for r in range(4):
            U[u_rows_b2[r]][c_global] = U[u_rows_b1[r]][c_offset]
        # Copy V
        for r in range(4):
            V[v_rows_b2[r]][c_global] = V[v_rows_b1[r]][c_offset]
        # Copy W
        for r in range(4):
            W[w_rows_b2[r]][c_global] = W[w_rows_b1[r]][c_offset]

    # --- Block 3: M_2,2,1 (Schoolbook) for C col 4 from B rows 0,1 col 4 and A cols 0,1 ---
    u_rows_b3 = u_rows_b1
    
    # V rows for B rows 0,1 col 4
    v_rows_b3 = [4, 9] # b04, b14
    
    # W rows for C col 4
    w_rows_b3 = [8, 9] # c40, c41
    
    # 4 terms
    # Term 0: a00 * b04 -> c40
    set_val(U, u_rows_b3[0], 14, 1)
    set_val(V, v_rows_b3[0], 14, 1)
    set_val(W, w_rows_b3[0], 14, 1)
    
    # Term 1: a01 * b14 -> c40
    set_val(U, u_rows_b3[1], 15, 1)
    set_val(V, v_rows_b3[1], 15, 1)
    set_val(W, w_rows_b3[0], 15, 1)
    
    # Term 2: a10 * b04 -> c41
    set_val(U, u_rows_b3[2], 16, 1)
    set_val(V, v_rows_b3[0], 16, 1)
    set_val(W, w_rows_b3[1], 16, 1)
    
    # Term 3: a11 * b14 -> c41
    set_val(U, u_rows_b3[3], 17, 1)
    set_val(V, v_rows_b3[1], 17, 1)
    set_val(W, w_rows_b3[1], 17, 1)

    # --- Block 4: M_2,1,5 (Schoolbook) for C cols 0-4 from B row 2 cols 0-4 and A col 2 ---
    # U rows for A col 2
    u_rows_b4 = [2, 5] # a02, a12
    
    # V rows for B row 2 cols 0-4
    v_rows_b4 = [10, 11, 12, 13, 14] # b20..b24
    
    # W rows for C cols 0-4
    w_rows_b4 = list(range(10)) # c00..c14
    
    # 10 terms
    for i in range(2):
        for k in range(5):
            col_idx = 18 + i * 5 + k
            u_row = u_rows_b4[i]
            v_row = v_rows_b4[k]
            w_row = w_rows_b4[k * 2 + i]
            
            set_val(U, u_row, col_idx, 1)
            set_val(V, v_row, col_idx, 1)
            set_val(W, w_row, col_idx, 1)

    return {"U": U, "V": V, "W": W}
