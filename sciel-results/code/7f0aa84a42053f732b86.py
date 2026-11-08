"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 decomposition for 2x2x2 matrix multiplication
    # Indices: a, b, c in {0, 1}
    # U row: a*2 + b
    # V row: b*2 + c
    # W row: c*2 + a
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0: M1 = (A00 + A11)(B00 + B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1: M2 = (A10 + A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C10
    
    # r=2: M3 = A00(B01 - B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # r=3: M4 = A11(B10 - B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # r=4: M5 = (A00 + A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00, C01
    
    # r=5: M6 = (A10 - A00)(B00 + B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11
    
    # r=6: M7 = (A01 - A11)(B10 + B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1 # C00, C11

    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            # Process U
            for row1 in range(4):
                val1 = U2[row1][r1]
                if val1 == 0: continue
                a = row1 // 2
                b = row1 % 2
                for row2 in range(4):
                    val2 = U2[row2][r2]
                    if val2 == 0: continue
                    a2 = row2 // 2
                    b2 = row2 % 2
                    # i = 2*a + a2, j = 2*b + b2
                    # index = i*4 + j = (2*a+a2)*4 + (2*b+b2) = 8*a + 4*a2 + 4*b + 2*b2
                    idx = 8*a + 4*a2 + 4*b + 2*b2
                    U[idx][r] = val1 * val2
            
            # Process V
            for row1 in range(4):
                val1 = V2[row1][r1]
                if val1 == 0: continue
                b = row1 // 2
                c = row1 % 2
                for row2 in range(4):
                    val2 = V2[row2][r2]
                    if val2 == 0: continue
                    b2 = row2 // 2
                    c2 = row2 % 2
                    # j = 2*b + b2, k = 2*c + c2
                    # index = j*4 + k = (2*b+b2)*4 + (2*c+c2) = 8*b + 4*b2 + 4*c + 2*c2
                    idx = 8*b + 4*b2 + 4*c + 2*c2
                    V[idx][r] = val1 * val2
            
            # Process W
            for row1 in range(4):
                val1 = W2[row1][r1]
                if val1 == 0: continue
                c = row1 // 2
                a = row1 % 2
                for row2 in range(4):
                    val2 = W2[row2][r2]
                    if val2 == 0: continue
                    c2 = row2 // 2
                    a2 = row2 % 2
                    # k = 2*c + c2, i = 2*a + a2
                    # index = k*4 + i = (2*c+c2)*4 + (2*a+a2) = 8*c + 4*c2 + 4*a + 2*a2
                    idx = 8*c + 4*c2 + 4*a + 2*a2
                    W[idx][r] = val1 * val2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
