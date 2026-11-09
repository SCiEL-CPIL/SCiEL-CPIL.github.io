import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,3,10} into M_{2,2,10} + M_{2,1,10}
    # M_{2,1,10} has rank 20 (schoolbook)
    # M_{2,2,10} has rank 35 (Winograd 2x2x2 x 1x1x5)
    
    # 1. Construct M_{2,1,10} factors
    # n=2, m=1, p=10
    n1, m1, p1 = 2, 1, 10
    rank1 = n1 * m1 * p1 # 20
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
                
    # 2. Construct M_{2,2,2} factors (Winograd)
    # n=2, m=2, p=2
    n2, m2, p2 = 2, 2, 2
    rank222 = 7
    U222 = [[0] * rank222 for _ in range(n2 * m2)]
    V222 = [[0] * rank222 for _ in range(m2 * p2)]
    W222 = [[0] * rank222 for _ in range(p2 * n2)]
    
    # Winograd factors for 2x2
    # U (A matrix coefficients)
    # Row 0: a00, Row 1: a01, Row 2: a10, Row 3: a11
    U222[0][0] = 1; U222[0][2] = 1; U222[0][3] = 1; U222[0][5] = 1
    U222[1][1] = 1; U222[1][2] = 1; U222[1][4] = 1; U222[1][6] = 1
    U222[2][3] = 1; U222[2][6] = 1
    U222[3][4] = 1; U222[3][5] = 1
    
    # V (B matrix coefficients)
    # Row 0: b00, Row 1: b01, Row 2: b10, Row 3: b11
    V222[0][0] = 1; V222[0][2] = 1; V222[0][3] = 1; V222[0][5] = 1
    V222[1][1] = 1; V222[1][2] = 1; V222[1][4] = 1; V222[1][6] = 1
    V222[2][3] = 1; V222[2][6] = 1
    V222[3][4] = 1; V222[3][5] = 1
    
    # W (C matrix coefficients)
    # Row 0: c00, Row 1: c01, Row 2: c10, Row 3: c11
    # c00 = m0 - m2 + m4 - m6
    W222[0][0] = 1; W222[0][2] = -1; W222[0][4] = 1; W222[0][6] = -1
    # c01 = m1 - m3 + m5 - m6
    W222[1][1] = 1; W222[1][3] = -1; W222[1][5] = 1; W222[1][6] = -1
    # c10 = m0 - m1 + m3 + m4
    W222[2][0] = 1; W222[2][1] = -1; W222[2][3] = 1; W222[2][4] = 1
    # c11 = m1 - m2 - m4 + m5
    W222[3][1] = 1; W222[3][2] = -1; W222[3][4] = -1; W222[3][5] = 1

    # 3. Construct M_{1,1,5} factors (Schoolbook)
    # n=1, m=1, p=5
    n3, m3, p3 = 1, 1, 5
    rank115 = n3 * m3 * p3 # 5
    U115 = [[0] * rank115 for _ in range(n3 * m3)]
    V115 = [[0] * rank115 for _ in range(m3 * p3)]
    W115 = [[0] * rank115 for _ in range(p3 * n3)]
    
    for k in range(p3):
        U115[0][k] = 1
        V115[k][k] = 1
        W115[k][k] = 1

    # 4. Combine M_{2,2,2} and M_{1,1,5} to get M_{2,2,10}
    # Rank = 7 * 5 = 35
    rank2210 = rank222 * rank115
    U2210 = [[0] * rank2210 for _ in range(n2 * m2)]
    V2210 = [[0] * rank2210 for _ in range(m2 * p1)] # p1=10
    W2210 = [[0] * rank2210 for _ in range(p1 * n2)]
    
    # Map indices
    # U2210[i*m2+j, r222*5 + r115] = U222[i*m2+j, r222] * U115[0, r115]
    # V2210[j*p1+k, r222*5 + r115] = V222[j*p2+k%2, r222] * V115[k//2, r115]
    # W2210[k*n2+i, r222*5 + r115] = W222[k%2*n2+i, r222] * W115[k//2, r115]
    
    for r222 in range(rank222):
        for r115 in range(rank115):
            r = r222 * rank115 + r115
            
            # U
            for i in range(n2):
                for j in range(m2):
                    U2210[i * m2 + j][r] = U222[i * m2 + j][r222] * U115[0][r115]
                    
            # V
            for j in range(m2):
                for k in range(p1):
                    k2 = k % 2
                    k5 = k // 2
                    V2210[j * p1 + k][r] = V222[j * p2 + k2][r222] * V115[k5][r115]
                    
            # W
            for k in range(p1):
                for i in range(n2):
                    k2 = k % 2
                    k5 = k // 2
                    W2210[k * n2 + i][r] = W222[k2 * n2 + i][r222] * W115[k5][r115]

    # 5. Combine M_{2,2,10} and M_{2,1,10} into M_{2,3,10}
    total_rank = rank2210 + rank1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy M_{2,2,10} (j=0,1)
    for i in range(n2):
        for j in range(m2):
            for r in range(rank2210):
                U[i * m + j][r] = U2210[i * m2 + j][r]
                
    for j in range(m2):
        for k in range(p1):
            for r in range(rank2210):
                V[j * p + k][r] = V2210[j * p1 + k][r]
                
    for k in range(p1):
        for i in range(n2):
            for r in range(rank2210):
                W[k * n + i][r] = W2210[k * n2 + i][r]

    # Copy M_{2,1,10} (j=2)
    offset = rank2210
    for i in range(n1):
        for j in range(m1):
            for r in range(rank1):
                U[i * m + (j + m2)][offset + r] = U1[i * m1 + j][r]
                
    for j in range(m1):
        for k in range(p1):
            for r in range(rank1):
                V[(j + m2) * p + k][offset + r] = V1[j * p1 + k][r]
                
    for k in range(p1):
        for i in range(n1):
            for r in range(rank1):
                W[k * n + i][offset + r] = W1[k * n1 + i][r]

    return {"U": U, "V": V, "W": W}
