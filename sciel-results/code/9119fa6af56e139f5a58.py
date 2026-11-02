import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_222():
    # Rank 7 decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U corresponds to A entries: a(0), b(1), c(2), d(3)
    # V corresponds to B entries: e(0), f(1), g(2), h(3)
    # W corresponds to C entries: c11(0), c12(1), c21(2), c22(3)
    
    # M1: (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def rank_223():
    # 2x2x3, rank 11
    # Split B into B1 (2x1) and B2 (2x2)
    # A B1: schoolbook 2x2x1 (rank 4)
    # A B2: strassen 2x2x2 (rank 7)
    U1, V1, W1 = schoolbook(2, 2, 1)
    U2, V2, W2 = strassen_222()
    
    U = []
    V = []
    W = []
    for i in range(4):
        U.append(U1[i] + [0]*7)
    for i in range(4):
        V.append(V1[i] + [0]*7)
    for i in range(4):
        W.append(W1[i] + [0]*7)
        
    for i in range(4):
        U.append([0]*4 + U2[i])
    for i in range(4):
        # V indices: B1 is col 0, B2 is cols 1,2 (mapped to 0,1 of strassen)
        # B2 is 2x2. Strassen V expects 2x2.
        # B total is 2x3. B1 is col 0. B2 is cols 1,2.
        # V1 has 4 rows (2x2? No, 2x1 is 2 rows? No, m*p = 2*1=2 rows? No.
        # schoolbook(2,2,1): U is 4x4, V is 2x4, W is 2x4.
        # Wait, V shape is (m*p, rank) = (2, 4).
        # strassen_222: V shape is (4, 7).
        # We need to align V rows.
        # B1 rows: 0, 1.
        # B2 rows: 0, 1 (local). Global rows 1, 2? No.
        # B is 2x3. Rows 0, 1.
        # B1 is cols 0. B2 is cols 1, 2.
        # V1 row 0 corresponds to B[0,0]. V1 row 1 corresponds to B[1,0].
        # V2 row 0 corresponds to B2[0,0] which is B[0,1].
        # V2 row 1 corresponds to B2[0,1] which is B[0,2].
        # V2 row 2 corresponds to B2[1,0] which is B[1,1].
        # V2 row 3 corresponds to B2[1,1] which is B[1,2].
        
        # So we need to map V2 rows to global V rows.
        # Global V has 6 rows (2x3).
        # V1 uses rows 0, 1.
        # V2 uses rows 1, 2, 3, 4? No.
        # B[0,1] is row 0, col 1. Index 0*3+1 = 1.
        # B[0,2] is row 0, col 2. Index 0*3+2 = 2.
        # B[1,1] is row 1, col 1. Index 1*3+1 = 4.
        # B[1,2] is row 1, col 2. Index 1*3+2 = 5.
        
        # V1 (2x4) -> rows 0, 1 of global V (6x11)
        # V2 (4x7) -> rows 1, 2, 4, 5 of global V
        
        row_map = [1, 2, 4, 5]
        new_row = [0]*11
        for local_r, global_r in enumerate(row_map):
             for c in range(7):
                 new_row[4+c] = V2[local_r][c]
        # Wait, V1 is 2x4. V2 is 4x7.
        # We are building V row by row.
        # V is 6x11.
        # For global row 0: V1[0] + 0
        # For global row 1: V1[1] + V2[0]
        # For global row 2: 0 + V2[1]
        # For global row 3: 0
        # For global row 4: 0 + V2[2]
        # For global row 5: 0 + V2[3]
        
        if i == 0:
            V.append(V1[0] + [0]*7)
        elif i == 1:
            V.append(V1[1] + V2[0])
        elif i == 2:
            V.append([0]*4 + V2[1])
        elif i == 3:
            V.append([0]*11)
        elif i == 4:
            V.append([0]*4 + V2[2])
        elif i == 5:
            V.append([0]*4 + V2[3])
            
    # W is 6x11 (3x2 output? No, 2x3 output is 6 entries).
    # W1 is 2x4. W2 is 4x7.
    # C1 is 2x1. C2 is 2x2.
    # C total is 2x3.
    # C11 (0,0) is from W1[0].
    # C21 (1,0) is from W1[1].
    # C12 (0,1) is from W2[1] (C12 of strassen).
    # C13 (0,2) is from W2[2] (C21 of strassen? No).
    # Strassen W: c11(0), c12(1), c21(2), c22(3).
    # C2 is 2x2.
    # C2[0,0] is C12? No.
    # B2 is cols 1,2.
    # C2[0,0] corresponds to C[0,1].
    # C2[0,1] corresponds to C[0,2].
    # C2[1,0] corresponds to C[1,1].
    # C2[1,1] corresponds to C[1,2].
    
    # Strassen output W2 indices:
    # 0: c11 (row 0, col 0 of C2) -> C[0,1]
    # 1: c12 (row 0, col 1 of C2) -> C[0,2]
    # 2: c21 (row 1, col 0 of C2) -> C[1,1]
    # 3: c22 (row 1, col 1 of C2) -> C[1,2]
    
    # Global W rows (6):
    # 0: C[0,0]
    # 1: C[0,1]
    # 2: C[0,2]
    # 3: C[1,0]
    # 4: C[1,1]
    # 5: C[1,2]
    
    # W1 rows (2):
    # 0: C[0,0]
    # 1: C[1,0]
    
    # So:
    # Global W 0: W1[0] + 0
    # Global W 1: 0 + W2[0]
    # Global W 2: 0 + W2[1]
    # Global W 3: W1[1] + 0
    # Global W 4: 0 + W2[2]
    # Global W 5: 0 + W2[3]
    
    W.append(W1[0] + [0]*7)
    W.append([0]*4 + W2[0])
    W.append([0]*4 + W2[1])
    W.append(W1[1] + [0]*7)
    W.append([0]*4 + W2[2])
    W.append([0]*4 + W2[3])
    
    return U, V, W

def rank_123():
    # 1x2x3, rank 6
    return schoolbook(1, 2, 3)

def rank_226():
    # 2x2x6, rank 21
    # Split B into three 2x2 blocks.
    # Each uses Strassen.
    U_blocks = []
    V_blocks = []
    W_blocks = []
    for _ in range(3):
        U, V, W = strassen_222()
        U_blocks.append(U)
        V_blocks.append(V)
        W_blocks.append(W)
        
    U = []
    V = []
    W = []
    for i in range(4):
        U.append(U_blocks[0][i] + U_blocks[1][i] + U_blocks[2][i])
        
    # V is 12x21 (2x6)
    # V1 rows 0,1 (cols 0,1)
    # V2 rows 2,3 (cols 2,3) -- Wait, 2x2 block is 4 rows? No.
    # Strassen V is 4x7.
    # Block 1: cols 0,1. Rows 0,1.
    # Block 2: cols 2,3. Rows 2,3? No.
    # B is 2x6.
    # Block 1: B[:, 0:2]. Rows 0,1.
    # Block 2: B[:, 2:4]. Rows 0,1.
    # Block 3: B[:, 4:6]. Rows 0,1.
    
    # Strassen V rows correspond to B entries.
    # B[0,0], B[0,1], B[1,0], B[1,1].
    # Global V rows:
    # 0: B[0,0]
    # 1: B[0,1]
    # 2: B[0,2]
    # 3: B[0,3]
    # 4: B[0,4]
    # 5: B[0,5]
    # 6: B[1,0]
    # 7: B[1,1]
    # 8: B[1,2]
    # 9: B[1,3]
    # 10: B[1,4]
    # 11: B[1,5]
    
    # V1 (4x7) maps to rows 0,1,6,7.
    # V2 (4x7) maps to rows 2,3,8,9.
    # V3 (4x7) maps to rows 4,5,10,11.
    
    for i in range(12):
        row = [0]*21
        if i == 0: row[0:7] = V_blocks[0][0]
        elif i == 1: row[0:7] = V_blocks[0][1]
        elif i == 2: row[7:14] = V_blocks[1][0]
        elif i == 3: row[7:14] = V_blocks[1][1]
        elif i == 4: row[14:21] = V_blocks[2][0]
        elif i == 5: row[14:21] = V_blocks[2][1]
        elif i == 6: row[0:7] = V_blocks[0][2]
        elif i == 7: row[0:7] = V_blocks[0][3]
        elif i == 8: row[7:14] = V_blocks[1][2]
        elif i == 9: row[7:14] = V_blocks[1][3]
        elif i == 10: row[14:21] = V_blocks[2][2]
        elif i == 11: row[14:21] = V_blocks[2][3]
        V.append(row)
        
    # W is 12x21 (2x6)
    # W1 (4x7) maps to cols 0,1 -> rows 0,1,6,7? No.
    # C is 2x6.
    # C[:, 0:2] from W1.
    # C[:, 2:4] from W2.
    # C[:, 4:6] from W3.
    
    for i in range(12):
        row = [0]*21
        if i == 0: row[0:7] = W_blocks[0][0]
        elif i == 1: row[0:7] = W_blocks[0][1]
        elif i == 2: row[7:14] = W_blocks[1][0]
        elif i == 3: row[7:14] = W_blocks[1][1]
        elif i == 4: row[14:21] = W_blocks[2][0]
        elif i == 5: row[14:21] = W_blocks[2][1]
        elif i == 6: row[0:7] = W_blocks[0][2]
        elif i == 7: row[0:7] = W_blocks[0][3]
        elif i == 8: row[7:14] = W_blocks[1][2]
        elif i == 9: row[7:14] = W_blocks[1][3]
        elif i == 10: row[14:21] = W_blocks[2][2]
        elif i == 11: row[14:21] = W_blocks[2][3]
        W.append(row)
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # A: 3x4, B: 4x7, C: 3x7
    # A11 (1x2), A12 (1x2), A21 (2x2), A22 (2x2)
    # B11 (2x1), B1_23 (2x6), B21 (2x1), B2_23 (2x6)
    
    # Blocks
    # 1. C11 = A11 B11 (1x2x1)
    U1, V1, W1 = schoolbook(1, 2, 1)
    
    # 2. C1_23 = A11 B1_23 + A12 B2_23 (1x2x6 + 1x2x6)
    U2a, V2a, W2a = schoolbook(1, 2, 6)
    U2b, V2b, W2b = schoolbook(1, 2, 6)
    
    # 3. C21 = A21 B11 + A22 B21 (2x2x1 + 2x2x1)
    U3a, V3a, W3a = schoolbook(2, 2, 1)
    U3b, V3b, W3b = schoolbook(2, 2, 1)
    
    # 4. C2_23 = A21 B1_23 + A22 B2_23 (2x2x6 + 2x2x6)
    U4a, V4a, W4a = rank_226()
    U4b, V4b, W4b = rank_226()
    
    # Combine
    rank = 2 + 12 + 12 + 4 + 4 + 21 + 21
    U = [[0]*rank for _ in range(12)] # 3x4
    V = [[0]*rank for _ in range(28)] # 4x7
    W = [[0]*rank for _ in range(21)] # 7x3
    
    # Mapping
    # A indices: i*m+j
    # A11: i=0, j=0,1 -> 0,1
    # A12: i=0, j=2,3 -> 2,3
    # A21: i=1,2, j=0,1 -> 4,5,8,9
    # A22: i=1,2, j=2,3 -> 6,7,10,11
    
    # B indices: j*p+k
    # B11: j=0,1, k=0 -> 0, 7
    # B1_23: j=0,1, k=1..6 -> 1,2,3,4,5,6, 8,9,10,11,12,13
    # B21: j=2,3, k=0 -> 14, 21
    # B2_23: j=2,3, k=1..6 -> 15..20, 22..27
    
    # C indices: k*n+i
    # C11: k=0, i=0 -> 0
    # C1_23: k=1..6, i=0 -> 1, 3, 5, 7, 9, 11
    # C21: k=0, i=1,2 -> 1, 2
    # C2_23: k=1..6, i=1,2 -> 4,5, 6,7, 8,9, 10,11, 12,13, 14,15
    
    # Block 1: C11
    for i in range(2): # U1 rows
        U[i][0] += U1[i][0]
        U[i][1] += U1[i][1]
    for i in range(2): # V1 rows (0, 7)
        V[i][0] += V1[i][0]
        V[i][1] += V1[i][1]
        V[i+7][0] += V1[i][0]
        V[i+7][1] += V1[i][1]
    for i in range(2): # W1 rows (0, 1)
        W[i][0] += W1[i][0]
        W[i][1] += W1[i][1]
        
    # Block 2: C1_23
    # A11 (rows 0,1) * B1_23
    for i in range(2):
        for c in range(12):
            U[i][2+c] += U2a[i][c]
    # B1_23 rows: 1..6, 8..13
    v_rows_1 = [1,2,3,4,5,6, 8,9,10,11,12,13]
    for i in range(12):
        for c in range(12):
            V[v_rows_1[i]][2+c] += V2a[i][c]
    # C1_23 rows: 1, 3, 5, 7, 9, 11
    w_rows_1 = [1, 3, 5, 7, 9, 11]
    for i in range(6):
        for c in range(12):
            W[w_rows_1[i]][2+c] += W2a[i][c]
            
    # A12 (rows 2,3) * B2_23
    for i in range(2):
        for c in range(12):
            U[i+2][14+c] += U2b[i][c]
    # B2_23 rows: 15..20, 22..27
    v_rows_2 = [15,16,17,18,19,20, 22,23,24,25,26,27]
    for i in range(12):
        for c in range(12):
            V[v_rows_2[i]][14+c] += V2b[i][c]
    for i in range(6):
        for c in range(12):
            W[w_rows_1[i]][14+c] += W2b[i][c]
            
    # Block 3: C21
    # A21 (rows 4,5,8,9) * B11
    for i in range(4):
        for c in range(4):
            U[i+4][26+c] += U3a[i][c]
    for i in range(2):
        for c in range(4):
            V[i][26+c] += V3a[i][c]
            V[i+7][26+c] += V3a[i][c]
    # C21 rows: 1, 2
    for i in range(2):
        for c in range(4):
            W[i+1][26+c] += W3a[i][c]
            
    # A22 (rows 6,7,10,11) * B21
    for i in range(4):
        for c in range(4):
            U[i+6][30+c] += U3b[i][c]
    for i in range(2):
        for c in range(4):
            V[i+14][30+c] += V3b[i][c]
            V[i+21][30+c] += V3b[i][c]
    for i in range(2):
        for c in range(4):
            W[i+1][30+c] += W3b[i][c]
            
    # Block 4: C2_23
    # A21 * B1_23
    for i in range(4):
        for c in range(21):
            U[i+4][34+c] += U4a[i][c]
    for i in range(12):
        for c in range(21):
            V[v_rows_1[i]][34+c] += V4a[i][c]
    # C2_23 rows: 4,5, 6,7, 8,9, 10,11, 12,13, 14,15
    w_rows_2 = [4,5, 6,7, 8,9, 10,11, 12,13, 14,15]
    for i in range(12):
        for c in range(21):
            W[w_rows_2[i]][34+c] += W4a[i][c]
            
    # A22 * B2_23
    for i in range(4):
        for c in range(21):
            U[i+6][55+c] += U4b[i][c]
    for i in range(12):
        for c in range(21):
            V[v_rows_2[i]][55+c] += V4b[i][c]
    for i in range(12):
        for c in range(21):
            W[w_rows_2[i]][55+c] += W4b[i][c]
            
    return {"U": U, "V": V, "W": W}

