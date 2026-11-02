"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = 2, 3, 10
    
    # Helper: build 2x2x2 Strassen decomposition
    # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A00+A11, B00+B11
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[2][0] = 1  # C00, C11
        
        # M2: A10+A11, B00
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1               # B00
        W[2][1] = 1; W[3][1] = 1  # C10, C11
        
        # M3: A00, B01-B11
        U[0][2] = 1               # A00
        V[1][2] = 1; V[3][2] = -1 # B01, -B11
        W[1][2] = 1; W[3][2] = 1  # C01, C11
        
        # M4: A11, B10-B00
        U[3][3] = 1               # A11
        V[2][3] = 1; V[0][3] = -1 # B10, -B00
        W[0][3] = 1; W[2][3] = 1  # C00, C10
        
        # M5: A00+A01, B11
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1               # B11
        W[0][4] = -1; W[1][4] = 1 # -C00, C01
        
        # M6: A10-A00, B00+B01
        U[2][5] = 1; U[0][5] = -1 # A10, -A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        W[3][5] = 1               # C11
        
        # M7: A01-A11, B10+B11
        U[1][6] = 1; U[3][6] = -1 # A01, -A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        W[0][6] = 1               # C00
        
        return U, V, W
    
    # Helper: build 2x1x2 decomposition (rank 4)
    # U: (2*1, 4), V: (1*2, 4), W: (2*2, 4)
    # Indexing: U[i*1+j, c]=U[i, c], V[j*2+k, c]=V[k, c], W[k*2+i, c]
    def trivial_2x1x2():
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(2)]
        W = [[0]*4 for _ in range(4)]
        # C00 = A00*B0, C01 = A00*B1, C10 = A10*B0, C11 = A10*B1
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # C00
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1  # C01
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1  # C10
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1  # C11
        return U, V, W
    
    # Build 2x3x3 decomposition
    # Split into 2x2x2 (cols 0,1 of B) and 2x1x2 (col 2 of B)
    def build_2x3x3():
        U_sv, V_sv, W_sv = strassen_2x2x2()  # 7 terms for cols 0,1
        U_tv, V_tv, W_tv = trivial_2x1x2()    # 4 terms for col 2
        
        rank = 11
        U = [[0]*rank for _ in range(6)]  # 2*3
        V = [[0]*rank for _ in range(6)]  # 3*3
        W = [[0]*rank for _ in range(4)]  # 3*2
        
        # Copy Strassen part (cols 0,1 of B)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3+j][c] = U_sv[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*3+k][c] = V_sv[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W_sv[k*2+i][c]
        
        # Copy trivial part (col 2 of B)
        offset = 7
        for c in range(4):
            cc = offset + c
            for i in range(2):
                U[i*3+2][cc] = U_tv[i][c]
            for k in range(2):
                V[2*3+k][cc] = V_tv[k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][cc] = W_tv[k*2+i][c]
        
        return U, V, W
    
    # Build 2x3x2 decomposition
    # Split A into 2x2 (cols 0,1) and 2x1 (col 2)
    def build_2x3x2():
        U_sv, V_sv, W_sv = strassen_2x2x2()  # 7 terms for A cols 0,1
        U_tv, V_tv, W_tv = trivial_2x1x2()    # 4 terms for A col 2
        
        rank = 11
        U = [[0]*rank for _ in range(6)]  # 2*3
        V = [[0]*rank for _ in range(6)]  # 3*2
        W = [[0]*rank for _ in range(4)]  # 2*2
        
        # Copy Strassen part (A cols 0,1)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3+j][c] = U_sv[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*2+k][c] = V_sv[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W_sv[k*2+i][c]
        
        # Copy trivial part (A col 2)
        offset = 7
        for c in range(4):
            cc = offset + c
            for i in range(2):
                U[i*3+2][cc] = U_tv[i][c]
            for k in range(2):
                V[2*2+k][cc] = V_tv[k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][cc] = W_tv[k*2+i][c]
        
        return U, V, W
    
    # Build 2x3x5 decomposition
    # Split B into 3x3 (cols 0,1,2) and 3x2 (cols 3,4)
    def build_2x3x5():
        U_3, V_3, W_3 = build_2x3x3()  # 11 terms
        U_2, V_2, W_2 = build_2x3x2()  # 11 terms
        
        rank = 22
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(15)]  # 3*5
        W = [[0]*rank for _ in range(10)]  # 5*2
        
        # Copy 2x3x3 part (B cols 0,1,2)
        for c in range(11):
            for i in range(2):
                for j in range(3):
                    U[i*3+j][c] = U_3[i*3+j][c]
            for j in range(3):
                for k in range(3):
                    V[j*5+k][c] = V_3[j*3+k][c]
            for k in range(3):
                for i in range(2):
                    W[k*2+i][c] = W_3[k*2+i][c]
        
        # Copy 2x3x2 part (B cols 3,4)
        offset = 11
        for c in range(11):
            cc = offset + c
            for i in range(2):
                for j in range(3):
                    U[i*3+j][cc] = U_2[i*3+j][c]
            for j in range(3):
                for k in range(2):
                    V[j*5+(3+k)][cc] = V_2[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[(3+k)*2+i][cc] = W_2[k*2+i][c]
        
        return U, V, W
    
    # Build 2x3x10 decomposition
    # Split B into two 3x5 blocks
    def build_2x3x10():
        U_5a, V_5a, W_5a = build_2x3x5()  # 22 terms, B cols 0-4
        U_5b, V_5b, W_5b = build_2x3x5()  # 22 terms, B cols 5-9
        
        rank = 44
        U = [[0]*rank for _ in range(6)]
        V = [[0]*rank for _ in range(30)]  # 3*10
        W = [[0]*rank for _ in range(20)]  # 10*2
        
        # Copy first 2x3x5 (B cols 0-4)
        for c in range(22):
            for i in range(2):
                for j in range(3):
                    U[i*3+j][c] = U_5a[i*3+j][c]
            for j in range(3):
                for k in range(5):
                    V[j*10+k][c] = V_5a[j*5+k][c]
            for k in range(5):
                for i in range(2):
                    W[k*2+i][c] = W_5a[k*2+i][c]
        
        # Copy second 2x3x5 (B cols 5-9)
        offset = 22
        for c in range(22):
            cc = offset + c
            for i in range(2):
                for j in range(3):
                    U[i*3+j][cc] = U_5b[i*3+j][c]
            for j in range(3):
                for k in range(5):
                    V[j*10+(5+k)][cc] = V_5b[j*5+k][c]
            for k in range(5):
                for i in range(2):
                    W[(5+k)*2+i][cc] = W_5b[k*2+i][c]
        
        return U, V, W
    
    return build_2x3x10()
