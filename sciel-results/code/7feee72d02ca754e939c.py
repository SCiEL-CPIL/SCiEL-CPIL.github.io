DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: A_L (2x2) * B_T1 (2x8) using Strassen
    # A_L indices in U: 0, 1, 3, 4
    # B_T1 indices in V: 0..7 (row 0), 9..16 (row 1)
    # C indices in W: k*2 + i
    
    for block in range(4):
        # B columns 2*block, 2*block+1
        b_col_0 = 2 * block
        b_col_1 = 2 * block + 1
        
        # V indices for B_T1 block
        v_idx_00 = 0 * 9 + b_col_0 # B[0][b_col_0]
        v_idx_01 = 0 * 9 + b_col_1 # B[0][b_col_1]
        v_idx_10 = 1 * 9 + b_col_0 # B[1][b_col_0]
        v_idx_11 = 1 * 9 + b_col_1 # B[1][b_col_1]
        
        # U indices for A_L
        u_idx_a = 0 # A[0][0]
        u_idx_b = 1 # A[0][1]
        u_idx_c = 3 # A[1][0]
        u_idx_d = 4 # A[1][1]
        
        # W indices for C block (2x2)
        w_idx_00 = b_col_0 * 2 + 0
        w_idx_01 = b_col_1 * 2 + 0
        w_idx_10 = b_col_0 * 2 + 1
        w_idx_11 = b_col_1 * 2 + 1
        
        # M1 = (a+d)(x+w)
        U[u_idx_a][col] += 1
        U[u_idx_d][col] += 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M2 = (c+d)x
        U[u_idx_c][col] += 1
        U[u_idx_d][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1
        W[w_idx_11][col] -= 1
        col += 1
        
        # M3 = a(y+w)
        U[u_idx_a][col] += 1
        V[v_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M4 = dw
        U[u_idx_d][col] += 1
        V[v_idx_10][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_10][col] += 1
        col += 1
        
        # M5 = (a+b)w
        U[u_idx_a][col] += 1
        U[u_idx_b][col] += 1
        V[v_idx_10][col] += 1
        W[w_idx_00][col] -= 1
        W[w_idx_01][col] += 1
        col += 1
        
        # M6 = (c-a)(x+y)
        U[u_idx_c][col] += 1
        U[u_idx_a][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M7 = (b-d)(x+w)
        U[u_idx_b][col] += 1
        U[u_idx_d][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    # Part 2: A_L (2x2) * B_T2 (2x1) using Schoolbook
    # B col 8
    b_col = 8
    v_idx_0 = 0 * 9 + b_col
    v_idx_1 = 1 * 9 + b_col
    
    w_idx_0 = b_col * 2 + 0
    w_idx_1 = b_col * 2 + 1
    
    # a*x
    U[0][col] = 1
    V[v_idx_0][col] = 1
    W[w_idx_0][col] = 1
    col += 1
    # b*w
    U[1][col] = 1
    V[v_idx_1][col] = 1
    W[w_idx_0][col] = 1
    col += 1
    # c*x
    U[3][col] = 1
    V[v_idx_0][col] = 1
    W[w_idx_1][col] = 1
    col += 1
    # d*w
    U[4][col] = 1
    V[v_idx_1][col] = 1
    W[w_idx_1][col] = 1
    col += 1

    # Part 3: A_R (2x1) * B_B (1x9) using Schoolbook
    # A col 2
    u_idx_a = 2
    u_idx_b = 5
    
    for k in range(9):
        v_idx = 2 * 9 + k
        
        w_idx_0 = k * 2 + 0
        w_idx_1 = k * 2 + 1
        
        # a*b
        U[u_idx_a][col] = 1
        V[v_idx][col] = 1
        W[w_idx_0][col] = 1
        col += 1
        
        # b*b
        U[u_idx_b][col] = 1
        V[v_idx][col] = 1
        W[w_idx_1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
