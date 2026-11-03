def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    
    # Rank 14 (7 * 2)
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 24 x 14
    W = [[0] * rank for _ in range(p * n)]  # 12 x 14
    
    # --- Part 1: Strassen 2x2x2 (Rank 7) ---
    # U1: 4x7, V1: 4x7, W1: 4x7
    # Indices: i1, j1, k1 in {0,1}
    # U1 row: i1*2 + j1
    # V1 row: j1*2 + k1
    # W1 row: k1*2 + i1
    
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # M2: (A10+A11)B00
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    # M3: A00(B01-B11)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    # M4: A11(B10-B00)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    # M5: (A00+A01)B11
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1

    # --- Part 2: 1x2x3 (Rank 2) ---
    # U2: 2x2 (rows j2), V2: 6x2 (rows j2*3+k2), W2: 3x2 (rows k2)
    U2 = [[0]*2 for _ in range(2)]
    V2 = [[0]*2 for _ in range(6)]
    W2 = [[0]*2 for _ in range(3)]
    
    # Term 1: A0 * B_row0
    U2[0][0] = 1 # picks A0 (index 0 in flattened A2)
    U2[1][0] = 0
    # V2 picks B00, B01, B02
    V2[0][0] = 1 # j2=0, k2=0
    V2[1][0] = 1 # j2=0, k2=1
    V2[2][0] = 1 # j2=0, k2=2
    # W2 distributes to C0, C1, C2
    W2[0][0] = 1
    W2[1][0] = 1
    W2[2][0] = 1
    
    # Term 2: A1 * B_row1
    U2[0][1] = 0
    U2[1][1] = 1 # picks A1 (index 1 in flattened A2)
    # V2 picks B10, B11, B12
    V2[3][1] = 1 # j2=1, k2=0
    V2[4][1] = 1 # j2=1, k2=1
    V2[5][1] = 1 # j2=1, k2=2
    # W2 distributes to C0, C1, C2
    W2[0][1] = 1
    W2[1][1] = 1
    W2[2][1] = 1

    # --- Combine ---
    # n1=2, m1=2, p1=2
    # n2=1, m2=2, p2=3
    # i = i1
    # j = j1*2 + j2
    # k = k1*3 + k2
    
    for c1 in range(7):
        for c2 in range(2):
            c = c1 * 2 + c2
            
            # Fill U
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1
                        j = j1 * 2 + j2
                        row_u = i * m + j
                        val = U1[i1 * 2 + j1][c1] * U2[j2][c2]
                        U[row_u][c] = val
                        
            # Fill V
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(3):
                            j = j1 * 2 + j2
                            k = k1 * 3 + k2
                            row_v = j * p + k
                            val = V1[j1 * 2 + k1][c1] * V2[j2 * 3 + k2][c2]
                            V[row_v][c] = val
                            
            # Fill W
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(3):
                        i = i1
                        k = k1 * 3 + k2
                        row_w = k * n + i
                        val = W1[k1 * 2 + i1][c1] * W2[k2][c2]
                        W[row_w][c] = val

    return {"U": U, "V": V, "W": W}
