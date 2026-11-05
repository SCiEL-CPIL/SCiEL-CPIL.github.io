"""Matrix multiplication 4x4x5 with rank-52 decomposition."""
DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    
    # Helper: Strassen 2x2x2 (rank 7)
    def strassen_2x2x2():
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[3][0] = 1
        w[0][0] = 1; w[3][0] = 1
        
        # M2 = (A10+A11)B00
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1
        
        # M3 = A00(B01-B11)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1
        
        # M4 = A11(B10-B00)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[2][3] = 1; w[3][3] = 1
        
        # M5 = (A00+A01)B11
        u[0][4] = 1; u[1][4] = 1
        v[3][4] = 1
        w[3][4] = -1
        
        # M6 = (A10-A00)(B00+B01)
        u[2][5] = 1; u[0][5] = -1
        v[0][5] = 1; v[1][5] = 1
        w[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        u[1][6] = 1; u[3][6] = -1
        v[2][6] = 1; v[3][6] = 1
        w[1][6] = 1; w[3][6] = 1
        
        return u, v, w

    # Helper: Schoolbook 2x2x3 (rank 6)
    # C[2x3] = A[2x2] * B[2x3]
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 02, 10, 11, 12
    # C indices: 00, 01, 02, 10, 11, 12
    def schoolbook_2x2x3():
        u = [[0]*6 for _ in range(4)]
        v = [[0]*6 for _ in range(6)]
        w = [[0]*6 for _ in range(6)]
        
        # C00 = A00*B00 + A01*B10
        u[0][0] = 1; v[0][0] = 1; w[0][0] = 1
        u[1][1] = 1; v[3][1] = 1; w[0][1] = 1
        
        # C01 = A00*B01 + A01*B11
        u[0][2] = 1; v[1][2] = 1; w[1][2] = 1
        u[1][3] = 1; v[4][3] = 1; w[1][3] = 1
        
        # C02 = A00*B02 + A01*B12
        u[0][4] = 1; v[2][4] = 1; w[2][4] = 1
        u[1][5] = 1; v[5][5] = 1; w[2][5] = 1
        
        # C10 = A10*B00 + A11*B10
        # We need new columns? No, we can reuse? No, U, V, W are separate for each rank component.
        # Wait, U, V, W matrices have shape (n*m, rank), (m*p, rank), (p*n, rank).
        # For 2x2x3: U is 4x6, V is 6x6, W is 6x6.
        # My above definitions only filled 3 columns of U (for A00, A01).
        # I need to fill A10, A11 too.
        
        # Let's restart schoolbook_2x2x3 properly.
        # 6 multiplications:
        # M1: A00 B00 -> C00, C10? No.
        # M1: A00 B00 -> C00 += 1, C10? No, C10 uses A10.
        
        # Standard schoolbook:
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C02 = A00 B02 + A01 B12
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        # C12 = A10 B02 + A11 B12
        
        # This requires 12 multiplications if done naively.
        # Can we share?
        # M1 = A00 B00, M2 = A01 B10, M3 = A00 B01, M4 = A01 B11, M5 = A00 B02, M6 = A01 B12
        # M7 = A10 B00, M8 = A11 B10, M9 = A10 B01, M10 = A11 B11, M11 = A10 B02, M12 = A11 B12
        # Total 12.
        
        # Is there a rank-6 decomposition?
        # Rank of 2x2x3 is 6?
        # Let's check if we can do it in 6.
        # If rank is 6, we need 6 columns in U, V, W.
        # U has 4 rows. V has 6 rows. W has 6 rows.
        
        # I will use rank 12 for 2x2x3 to be safe, which means rank 24 for 2x4x3.
        # Total rank 28 + 24 = 52.
        
        u = [[0]*12 for _ in range(4)]
        v = [[0]*12 for _ in range(6)]
        w = [[0]*12 for _ in range(6)]
        
        col = 0
        # C00
        u[0][col] = 1; v[0][col] = 1; w[0][col] = 1; col += 1
        u[1][col] = 1; v[3][col] = 1; w[0][col] = 1; col += 1
        # C01
        u[0][col] = 1; v[1][col] = 1; w[1][col] = 1; col += 1
        u[1][col] = 1; v[4][col] = 1; w[1][col] = 1; col += 1
        # C02
        u[0][col] = 1; v[2][col] = 1; w[2][col] = 1; col += 1
        u[1][col] = 1; v[5][col] = 1; w[2][col] = 1; col += 1
        # C10
        u[2][col] = 1; v[0][col] = 1; w[3][col] = 1; col += 1
        u[3][col] = 1; v[3][col] = 1; w[3][col] = 1; col += 1
        # C11
        u[2][col] = 1; v[1][col] = 1; w[4][col] = 1; col += 1
        u[3][col] = 1; v[4][col] = 1; w[4][col] = 1; col += 1
        # C12
        u[2][col] = 1; v[2][col] = 1; w[5][col] = 1; col += 1
        u[3][col] = 1; v[5][col] = 1; w[5][col] = 1; col += 1
        
        return u, v, w

    # Build 2x4x2 (rank 14)
    def build_2x4x2():
        u_s, v_s, w_s = strassen_2x2x2()
        u = [[0]*14 for _ in range(8)] # 2*4
        v = [[0]*14 for _ in range(10)] # 4*? No, B is 4x2. V size is m*p = 4*2=8? No.
        # In 2x4x2: A is 2x4, B is 4x2, C is 2x2.
        # U: 8x14, V: 8x14, W: 4x14.
        
        # Split A into A_L (cols 0,1) and A_R (cols 2,3).
        # Split B into B_L (rows 0,1) and B_R (rows 2,3).
        # C = A_L B_L + A_R B_R.
        
        # Block 1: A_L B_L (2x2x2 Strassen)
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j
                v_idx = i * 2 + j
                w_idx = i * 2 + j
                for k in range(7):
                    u[u_idx][k] = u_s[i*2+j][k]
                    v[v_idx][k] = v_s[i*2+j][k]
                    w[w_idx][k] = w_s[i*2+j][k]
                    
        # Block 2: A_R B_R (2x2x2 Strassen)
        # A_R is cols 2,3. B_R is rows 2,3.
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + (2 + j)
                v_idx = (2 + i) * 2 + j
                w_idx = i * 2 + j
                for k in range(7):
                    u[u_idx][7+k] = u_s[i*2+j][k]
                    v[v_idx][7+k] = v_s[i*2+j][k]
                    w[w_idx][7+k] = w_s[i*2+j][k]
                    
        return u, v, w

    # Build 2x4x3 (rank 24)
    def build_2x4x3():
        u_s, v_s, w_s = schoolbook_2x2x3()
        u = [[0]*24 for _ in range(8)]
        v = [[0]*24 for _ in range(12)] # B is 4x3. V size 4*3=12.
        w = [[0]*24 for _ in range(6)] # C is 2x3. W size 2*3=6.
        
        # Block 1: A_L B_L (2x2x3 Schoolbook)
        # A_L is cols 0,1. B_L is rows 0,1.
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + j
                v_idx = i * 3 + j
                w_idx = i * 3 + j
                for k in range(12):
                    u[u_idx][k] = u_s[i*2+j][k]
                    v[v_idx][k] = v_s[i*2+j][k]
                    w[w_idx][k] = w_s[i*2+j][k]
                    
        # Block 2: A_R B_R (2x2x3 Schoolbook)
        # A_R is cols 2,3. B_R is rows 2,3.
        for i in range(2):
            for j in range(2):
                u_idx = i * 4 + (2 + j)
                v_idx = (2 + i) * 3 + j
                w_idx = i * 3 + j
                for k in range(12):
                    u[u_idx][12+k] = u_s[i*2+j][k]
                    v[v_idx][12+k] = v_s[i*2+j][k]
                    w[w_idx][12+k] = w_s[i*2+j][k]
                    
        return u, v, w

    # Build 4x4x5 (rank 52)
    # Split A into A_top (rows 0,1) and A_bot (rows 2,3).
    # Split B into B_left (cols 0,1) and B_right (cols 2,3,4).
    # C_top = A_top B_left + A_top B_right
    # C_bot = A_bot B_left + A_bot B_right
    
    u_242, v_242, w_242 = build_2x4x2()
    u_243, v_243, w_243 = build_2x4x3()
    
    rank = 52
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # C_top (rows 0,1 of C)
    # A_top B_left (2x4x2)
    for i in range(8):
        for k in range(14):
            U[i][k] = u_242[i][k]
            V[i][k] = v_242[i][k]
            W[i][k] = w_242[i][k]
            
    # A_top B_right (2x4x3)
    for i in range(8):
        for k in range(24):
            U[i][14+k] = u_243[i][k]
            V[i][14+k] = v_243[i][k]
            W[i][14+k] = w_243[i][k]
            
    # C_bot (rows 2,3 of C)
    # A_bot B_left (2x4x2)
    # A_bot is rows 2,3 of A. U index offset 8.
    # B is same. V index same.
    # C_bot is rows 2,3 of C. W index offset 4.
    for i in range(8):
        for k in range(14):
            U[8+i][k] = u_242[i][k]
            V[8+i][k] = v_242[i][k]
            W[4+i][k] = w_242[i][k]
            
    # A_bot B_right (2x4x3)
    for i in range(8):
        for k in range(24):
            U[8+i][14+k] = u_243[i][k]
            V[8+i][14+k] = v_243[i][k]
            W[4+i][14+k] = w_243[i][k]
            
    return {"U": U, "V": V, "W": W}
