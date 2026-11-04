import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 48 decomposition
    # 4 blocks of 2x2x3 (rank 12 each)
    # Total rank 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
    # A indices: 0,1,5,6
    # B indices: 0,1,2, 6,7,8
    # C indices: 0,1,2, 6,7,8 (W is p*n = 12. W[k*n+i])
    # W[0,0]=0, W[0,1]=2, W[0,2]=4
    # W[1,0]=1, W[1,1]=3, W[1,2]=5
    
    # Col 0
    # 1. A00 B00
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # 2. A01 B10
    U[1][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    # 3. A10 B00
    U[5][col] = 1; V[0][col] = 1; W[1][col] = 1; col += 1
    # 4. A11 B10
    U[6][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    
    # Col 1
    # 1. A00 B01
    U[0][col] = 1; V[1][col] = 1; W[2][col] = 1; col += 1
    # 2. A01 B11
    U[1][col] = 1; V[7][col] = 1; W[2][col] = 1; col += 1
    # 3. A10 B01
    U[5][col] = 1; V[1][col] = 1; W[3][col] = 1; col += 1
    # 4. A11 B11
    U[6][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    
    # Col 2
    # 1. A00 B02
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    # 2. A01 B12
    U[1][col] = 1; V[8][col] = 1; W[4][col] = 1; col += 1
    # 3. A10 B02
    U[5][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    # 4. A11 B12
    U[6][col] = 1; V[8][col] = 1; W[5][col] = 1; col += 1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 3:6] -> C[0:2, 3:6]
    # A indices: 0,1,5,6
    # B indices: 0,1,2, 6,7,8 (rows 0,1. cols 3,4,5 -> 0*6+3=3, 1*6+3=9, etc)
    # B[0,3]=3, B[0,4]=4, B[0,5]=5
    # B[1,3]=9, B[1,4]=10, B[1,5]=11
    # C indices: 3,4,5, 9,10,11 (W[k*n+i])
    # W[3,0]=6, W[3,1]=8, W[3,2]=10
    # W[4,0]=7, W[4,1]=9, W[4,2]=11
    
    # Col 3
    # 1. A00 B03
    U[0][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    # 2. A01 B13
    U[1][col] = 1; V[9][col] = 1; W[6][col] = 1; col += 1
    # 3. A10 B03
    U[5][col] = 1; V[3][col] = 1; W[7][col] = 1; col += 1
    # 4. A11 B13
    U[6][col] = 1; V[9][col] = 1; W[7][col] = 1; col += 1
    
    # Col 4
    # 1. A00 B04
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    # 2. A01 B14
    U[1][col] = 1; V[10][col] = 1; W[8][col] = 1; col += 1
    # 3. A10 B04
    U[5][col] = 1; V[4][col] = 1; W[9][col] = 1; col += 1
    # 4. A11 B14
    U[6][col] = 1; V[10][col] = 1; W[9][col] = 1; col += 1
    
    # Col 5
    # 1. A00 B05
    U[0][col] = 1; V[5][col] = 1; W[10][col] = 1; col += 1
    # 2. A01 B15
    U[1][col] = 1; V[11][col] = 1; W[10][col] = 1; col += 1
    # 3. A10 B05
    U[5][col] = 1; V[5][col] = 1; W[11][col] = 1; col += 1
    # 4. A11 B15
    U[6][col] = 1; V[11][col] = 1; W[11][col] = 1; col += 1
    
    # Block 3: A[0:2, 2:5] * B[2:5, 0:3] -> C[0:2, 0:3]
    # A indices: 2,3,4, 7,8,9
    # B indices: 12,13,14, 18,19,20, 24,25,26
    # C indices: 0,1,2, 6,7,8
    
    # Col 0
    # 1. A02 B20
    U[2][col] = 1; V[12][col] = 1; W[0][col] = 1; col += 1
    # 2. A03 B30
    U[3][col] = 1; V[18][col] = 1; W[0][col] = 1; col += 1
    # 3. A04 B40
    U[4][col] = 1; V[24][col] = 1; W[0][col] = 1; col += 1
    # 4. A12 B20
    U[7][col] = 1; V[12][col] = 1; W[1][col] = 1; col += 1
    # 5. A13 B30
    U[8][col] = 1; V[18][col] = 1; W[1][col] = 1; col += 1
    # 6. A14 B40
    U[9][col] = 1; V[24][col] = 1; W[1][col] = 1; col += 1
    
    # Col 1
    # 1. A02 B21
    U[2][col] = 1; V[13][col] = 1; W[2][col] = 1; col += 1
    # 2. A03 B31
    U[3][col] = 1; V[19][col] = 1; W[2][col] = 1; col += 1
    # 3. A04 B41
    U[4][col] = 1; V[25][col] = 1; W[2][col] = 1; col += 1
    # 4. A12 B21
    U[7][col] = 1; V[13][col] = 1; W[3][col] = 1; col += 1
    # 5. A13 B31
    U[8][col] = 1; V[19][col] = 1; W[3][col] = 1; col += 1
    # 6. A14 B41
    U[9][col] = 1; V[25][col] = 1; W[3][col] = 1; col += 1
    
    # Col 2
    # 1. A02 B22
    U[2][col] = 1; V[14][col] = 1; W[4][col] = 1; col += 1
    # 2. A03 B32
    U[3][col] = 1; V[20][col] = 1; W[4][col] = 1; col += 1
    # 3. A04 B42
    U[4][col] = 1; V[26][col] = 1; W[4][col] = 1; col += 1
    # 4. A12 B22
    U[7][col] = 1; V[14][col] = 1; W[5][col] = 1; col += 1
    # 5. A13 B32
    U[8][col] = 1; V[20][col] = 1; W[5][col] = 1; col += 1
    # 6. A14 B42
    U[9][col] = 1; V[26][col] = 1; W[5][col] = 1; col += 1
    
    # Block 4: A[0:2, 2:5] * B[2:5, 3:6] -> C[0:2, 3:6]
    # A indices: 2,3,4, 7,8,9
    # B indices: 12,13,14, 18,19,20, 24,25,26 (rows 2,3,4. cols 3,4,5 -> 15,16,17, 21,22,23, 27,28,29)
    # C indices: 3,4,5, 9,10,11
    
    # Col 3
    # 1. A02 B23
    U[2][col] = 1; V[15][col] = 1; W[6][col] = 1; col += 1
    # 2. A03 B33
    U[3][col] = 1; V[21][col] = 1; W[6][col] = 1; col += 1
    # 3. A04 B43
    U[4][col] = 1; V[27][col] = 1; W[6][col] = 1; col += 1
    # 4. A12 B23
    U[7][col] = 1; V[15][col] = 1; W[7][col] = 1; col += 1
    # 5. A13 B33
    U[8][col] = 1; V[21][col] = 1; W[7][col] = 1; col += 1
    # 6. A14 B43
    U[9][col] = 1; V[27][col] = 1; W[7][col] = 1; col += 1
    
    # Col 4
    # 1. A02 B24
    U[2][col] = 1; V[16][col] = 1; W[8][col] = 1; col += 1
    # 2. A03 B34
    U[3][col] = 1; V[22][col] = 1; W[8][col] = 1; col += 1
    # 3. A04 B44
    U[4][col] = 1; V[28][col] = 1; W[8][col] = 1; col += 1
    # 4. A12 B24
    U[7][col] = 1; V[16][col] = 1; W[9][col] = 1; col += 1
    # 5. A13 B34
    U[8][col] = 1; V[22][col] = 1; W[9][col] = 1; col += 1
    # 6. A14 B44
    U[9][col] = 1; V[28][col] = 1; W[9][col] = 1; col += 1
    
    # Col 5
    # 1. A02 B25
    U[2][col] = 1; V[17][col] = 1; W[10][col] = 1; col += 1
    # 2. A03 B35
    U[3][col] = 1; V[23][col] = 1; W[10][col] = 1; col += 1
    # 3. A04 B45
    U[4][col] = 1; V[29][col] = 1; W[10][col] = 1; col += 1
    # 4. A12 B25
    U[7][col] = 1; V[17][col] = 1; W[11][col] = 1; col += 1
    # 5. A13 B35
    U[8][col] = 1; V[23][col] = 1; W[11][col] = 1; col += 1
    # 6. A14 B45
    U[9][col] = 1; V[29][col] = 1; W[11][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
