import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook factors for n x m by m x k
    def schoolbook(n, m, k):
        rank = n * m * k
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * k)]
        W = [[0] * rank for _ in range(k * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(k):
                    U[i * m + j][col] = 1
                    V[j * k + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen factors for 2 x 2 by 2 x 2
    def strassen_2x2():
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # c00 += 1, c11 += 1
        
        # M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # c10 += 1
        
        # M3 = a00(b01-b11)  (Corrected from previous thought)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1 # c01 += 1
        
        # M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1 # c00 += 1
        
        # M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # c00 -= 1, c01 += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # c11 += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1 # c00 -= 1
        
        return U, V, W

    # Split p=7 into 2, 2, 2, 1
    blocks = [2, 2, 2, 1]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for k in blocks:
        if k == 2:
            U_b, V_b, W_b = strassen_2x2()
        else:
            U_b, V_b, W_b = schoolbook(2, 2, k)
        
        U_parts.append(U_b)
        V_parts.append(V_b)
        W_parts.append(W_b)
        
        if time.time() > deadline:
            break

    # Concatenate
    # U is 4 x total_rank
    # V is 14 x total_rank (2x2 block has 4 rows, 2x1 block has 2 rows? No, V rows are m*k. For k=2, m=2, rows=4. For k=1, m=2, rows=2.)
    # Wait, V rows depend on k.
    # For k=2, V is 4x7.
    # For k=1, V is 2x4.
    # We need to align V rows.
    # B is 2x7. Rows of V correspond to B entries.
    # Block 1 (k=2): B cols 0,1. V rows 0,1,2,3 correspond to b00,b01,b10,b11.
    # Block 2 (k=2): B cols 2,3. V rows 4,5,6,7 correspond to b02,b03,b12,b13.
    # Block 3 (k=2): B cols 4,5. V rows 8,9,10,11 correspond to b04,b05,b14,b15.
    # Block 4 (k=1): B col 6. V rows 12,13 correspond to b06,b16.
    
    # Similarly for W.
    # W rows correspond to C entries.
    # Block 1 (k=2): C cols 0,1. W rows 0,1,2,3 correspond to c00,c01,c10,c11.
    # Block 2 (k=2): C cols 2,3. W rows 4,5,6,7 correspond to c02,c03,c12,c13.
    # Block 3 (k=2): C cols 4,5. W rows 8,9,10,11 correspond to c04,c05,c14,c15.
    # Block 4 (k=1): C col 6. W rows 12,13 correspond to c06,c16.
    
    # We need to construct full U, V, W.
    total_rank = sum(len(U_p[0]) for U_p in U_parts)
    
    U = [[0] * total_rank for _ in range(4)]
    V = [[0] * total_rank for _ in range(14)]
    W = [[0] * total_rank for _ in range(14)]
    
    col_offset = 0
    v_row_offset = 0
    w_row_offset = 0
    
    for i, k in enumerate(blocks):
        U_p = U_parts[i]
        V_p = V_parts[i]
        W_p = W_parts[i]
        rank_p = len(U_p[0])
        
        # Fill U
        for r in range(4):
            for c in range(rank_p):
                U[r][col_offset + c] = U_p[r][c]
        
        # Fill V
        # V_p has shape (2*k, rank_p)
        # Rows of V_p correspond to B block rows.
        # B block is 2 x k.
        # Row 0 of V_p -> B row 0, col 0..k-1
        # Row 1 of V_p -> B row 1, col 0..k-1
        # In global V, B row 0 cols v_row_offset..v_row_offset+k-1 are at v_row_offset + 0..k-1
        # B row 1 cols v_row_offset..v_row_offset+k-1 are at v_row_offset + k..v_row_offset+2k-1
        for r in range(2 * k):
            # r is local row in V_p.
            # global row = v_row_offset + r
            for c in range(rank_p):
                V[v_row_offset + r][col_offset + c] = V_p[r][c]
        
        # Fill W
        # W_p has shape (k*n, rank_p) = (2k, rank_p)
        # Rows of W_p correspond to C block rows.
        # C block is 2 x k.
        # Row 0 of W_p -> C row 0, col 0..k-1
        # Row 1 of W_p -> C row 1, col 0..k-1
        # In global W, C row 0 cols w_row_offset..w_row_offset+k-1 are at w_row_offset + 0..k-1
        # C row 1 cols w_row_offset..w_row_offset+k-1 are at w_row_offset + k..w_row_offset+2k-1
        for r in range(2 * k):
            for c in range(rank_p):
                W[w_row_offset + r][col_offset + c] = W_p[r][c]
        
        col_offset += rank_p
        v_row_offset += 2 * k
        w_row_offset += 2 * k
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
