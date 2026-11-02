"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decomposition: T_4,4,6 = T_2,2,2 (Strassen) x T_2,2,3 (Strassen + Schoolbook)
    # Rank 1: 7 (Strassen 2x2x2)
    # Rank 2: 11 (7 for 2x2x2 part, 4 for 2x2x1 part)
    # Total Rank: 7 * 11 = 77
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local 1: 2x2x2 Strassen
    # U1, V1, W1 are 4x7 matrices
    # Indices: i1, j1, k1 in {0, 1}
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # Strassen terms
    # 1: (a+d)(e+h)
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # 2: (c+d)e
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[2][1] = 1; W1[3][1] = -1 # C10+=, C11-=
    
    # 3: a(f-h)
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[1][2] = 1; W1[3][2] = 1 # C01+=, C11+=
    
    # 4: d(g-e)
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = -1; W1[2][3] = 1 # C00-=, C10+=
    
    # 5: (a+b)h
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[1][4] = 1 # C00-=, C01+=
    
    # 6: (c-a)(e+f)
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1 # C11+=
    
    # 7: (b-d)(g+h)
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1 # C00+=
    
    # Local 2: 2x2x3 (Rank 11)
    # U2, V2, W2
    # U2: 4x11 (i2, j2)
    # V2: 6x11 (j2, k2) where k2 in {0,1,2}
    # W2: 6x11 (k2, i2) where k2 in {0,1,2}
    U2 = [[0]*11 for _ in range(4)]
    V2 = [[0]*11 for _ in range(6)]
    W2 = [[0]*11 for _ in range(6)]
    
    # Part 1: Strassen for k2 in {0,1} (r2 0..6)
    # Map Strassen V1, W1 to V2, W2 with k2 offset 0
    for r in range(7):
        # U2 is same as U1
        for i2 in range(2):
            for j2 in range(2):
                idx_u2 = i2 * 2 + j2
                idx_u1 = i2 * 2 + j2
                U2[idx_u2][r] = U1[idx_u1][r]
                
        # V2: j2 in {0,1}, k2 in {0,1}
        for j2 in range(2):
            for k2 in range(2):
                idx_v2 = j2 * 3 + k2
                idx_v1 = j2 * 2 + k2
                V2[idx_v2][r] = V1[idx_v1][r]
                
        # W2: k2 in {0,1}, i2 in {0,1}
        for k2 in range(2):
            for i2 in range(2):
                idx_w2 = k2 * 2 + i2
                idx_w1 = k2 * 2 + i2
                W2[idx_w2][r] = W1[idx_w1][r]
                
    # Part 2: Schoolbook for k2=2 (r2 7..10)
    # Term 7: A00 B02 -> C02
    U2[0][7] = 1
    V2[0*3+2][7] = 1
    W2[2*2+0][7] = 1
    
    # Term 8: A01 B12 -> C02
    U2[1][8] = 1
    V2[1*3+2][8] = 1
    W2[2*2+0][8] = 1
    
    # Term 9: A10 B02 -> C12
    U2[2][9] = 1
    V2[0*3+2][9] = 1
    W2[2*2+1][9] = 1
    
    # Term 10: A11 B12 -> C12
    U2[3][10] = 1
    V2[1*3+2][10] = 1
    W2[2*2+1][10] = 1
    
    # Combine
    for r1 in range(7):
        for r2 in range(11):
            r = r1 * 11 + r2
            for i2 in range(2):
                for i1 in range(2):
                    i = 2 * i2 + i1
                    for j2 in range(2):
                        for j1 in range(2):
                            j = 2 * j2 + j1
                            U[i * m + j][r] = U1[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                            
            for j2 in range(2):
                for j1 in range(2):
                    j = 2 * j2 + j1
                    for k2 in range(3):
                        for k1 in range(2):
                            k = 2 * k2 + k1
                            V[j * p + k][r] = V1[j1 * 2 + k1][r1] * V2[j2 * 3 + k2][r2]
                            
            for k2 in range(3):
                for k1 in range(2):
                    k = 2 * k2 + k1
                    for i2 in range(2):
                        for i1 in range(2):
                            i = 2 * i2 + i1
                            W[k * n + i][r] = W1[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
