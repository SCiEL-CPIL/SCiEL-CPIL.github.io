import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 3, 6
    
    def make_schoolbook(n, m, p):
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
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Laderman 23-term decomposition for 3x3x3
    # Coefficients for U (9x23), V (9x23), W (9x23)
    # Indices: 0..22
    
    U_lad = [[0]*23 for _ in range(9)]
    V_lad = [[0]*23 for _ in range(9)]
    W_lad = [[0]*23 for _ in range(9)]
    
    # A = [a00 a01 a02; a10 a11 a12; a20 a21 a22]
    # B = [b00 b01 b02; b10 b11 b12; b20 b21 b22]
    # C = [c00 c01 c02; c10 c11 c12; c20 c21 c22]
    
    # 1. a00 b00 -> c00
    U_lad[0][0] = 1; V_lad[0][0] = 1; W_lad[0][0] = 1
    # 2. a01 b10 -> c01
    U_lad[1][1] = 1; V_lad[3][1] = 1; W_lad[1][1] = 1
    # 3. a02 b20 -> c02
    U_lad[2][2] = 1; V_lad[6][2] = 1; W_lad[2][2] = 1
    # 4. a10 b01 -> c10
    U_lad[3][3] = 1; V_lad[1][3] = 1; W_lad[3][3] = 1
    # 5. a11 b11 -> c11
    U_lad[4][4] = 1; V_lad[4][4] = 1; W_lad[4][4] = 1
    # 6. a12 b21 -> c12
    U_lad[5][5] = 1; V_lad[7][5] = 1; W_lad[5][5] = 1
    # 7. a20 b02 -> c20
    U_lad[6][6] = 1; V_lad[2][6] = 1; W_lad[6][6] = 1
    # 8. a21 b12 -> c21
    U_lad[7][7] = 1; V_lad[5][7] = 1; W_lad[7][7] = 1
    # 9. a22 b22 -> c22
    U_lad[8][8] = 1; V_lad[8][8] = 1; W_lad[8][8] = 1
    
    # 10. (a00+a11-a22)(b00+b11-b22)
    U_lad[0][9] = 1; U_lad[4][9] = 1; U_lad[8][9] = -1
    V_lad[0][9] = 1; V_lad[4][9] = 1; V_lad[8][9] = -1
    W_lad[0][9] = 1; W_lad[4][9] = 1; W_lad[8][9] = -1
    
    # 11. (a01+a12-a20)(b10+b21-b02)
    U_lad[1][10] = 1; U_lad[5][10] = 1; U_lad[6][10] = -1
    V_lad[3][10] = 1; V_lad[7][10] = 1; V_lad[2][10] = -1
    W_lad[1][10] = 1; W_lad[5][10] = 1; W_lad[6][10] = -1
    
    # 12. (a02+a20-a10)(b20+b02-b10)
    U_lad[2][11] = 1; U_lad[6][11] = 1; U_lad[3][11] = -1
    V_lad[6][11] = 1; V_lad[2][11] = 1; V_lad[3][11] = -1
    W_lad[2][11] = 1; W_lad[6][11] = 1; W_lad[3][11] = -1
    
    # 13. (a00+a01+a02)(b00+b01+b02)
    U_lad[0][12] = 1; U_lad[1][12] = 1; U_lad[2][12] = 1
    V_lad[0][12] = 1; V_lad[1][12] = 1; V_lad[2][12] = 1
    W_lad[0][12] = 1; W_lad[1][12] = 1; W_lad[2][12] = 1
    
    # 14. (a10+a11+a12)(b10+b11+b12)
    U_lad[3][13] = 1; U_lad[4][13] = 1; U_lad[5][13] = 1
    V_lad[3][13] = 1; V_lad[4][13] = 1; V_lad[5][13] = 1
    W_lad[3][13] = 1; W_lad[4][13] = 1; W_lad[5][13] = 1
    
    # 15. (a20+a21+a22)(b20+b21+b22)
    U_lad[6][14] = 1; U_lad[7][14] = 1; U_lad[8][14] = 1
    V_lad[6][14] = 1; V_lad[7][14] = 1; V_lad[8][14] = 1
    W_lad[6][14] = 1; W_lad[7][14] = 1; W_lad[8][14] = 1
    
    # 16. (a00+a10+a20)(b00+b10+b20)
    U_lad[0][15] = 1; U_lad[3][15] = 1; U_lad[6][15] = 1
    V_lad[0][15] = 1; V_lad[3][15] = 1; V_lad[6][15] = 1
    W_lad[0][15] = 1; W_lad[3][15] = 1; W_lad[6][15] = 1
    
    # 17. (a01+a11+a21)(b01+b11+b21)
    U_lad[1][16] = 1; U_lad[4][16] = 1; U_lad[7][16] = 1
    V_lad[1][16] = 1; V_lad[4][16] = 1; V_lad[7][16] = 1
    W_lad[1][16] = 1; W_lad[4][16] = 1; W_lad[7][16] = 1
    
    # 18. (a02+a12+a22)(b02+b12+b22)
    U_lad[2][17] = 1; U_lad[5][17] = 1; U_lad[8][17] = 1
    V_lad[2][17] = 1; V_lad[5][17] = 1; V_lad[8][17] = 1
    W_lad[2][17] = 1; W_lad[5][17] = 1; W_lad[8][17] = 1
    
    # 19. (a00+a01-a10-a11)(b00+b01-b10-b11)
    U_lad[0][18] = 1; U_lad[1][18] = 1; U_lad[3][18] = -1; U_lad[4][18] = -1
    V_lad[0][18] = 1; V_lad[1][18] = 1; V_lad[3][18] = -1; V_lad[4][18] = -1
    W_lad[0][18] = 1; W_lad[1][18] = 1; W_lad[3][18] = -1; W_lad[4][18] = -1
    
    # 20. (a01+a02-a11-a12)(b10+b11-b20-b21)
    U_lad[1][19] = 1; U_lad[2][19] = 1; U_lad[4][19] = -1; U_lad[5][19] = -1
    V_lad[3][19] = 1; V_lad[4][19] = 1; V_lad[6][19] = -1; V_lad[7][19] = -1
    W_lad[1][19] = 1; W_lad[2][19] = 1; W_lad[4][19] = -1; W_lad[5][19] = -1
    
    # 21. (a02+a00-a12-a10)(b20+b00-b21-b01)
    U_lad[2][20] = 1; U_lad[0][20] = 1; U_lad[5][20] = -1; U_lad[3][20] = -1
    V_lad[6][20] = 1; V_lad[0][20] = 1; V_lad[7][20] = -1; V_lad[1][20] = -1
    W_lad[2][20] = 1; W_lad[0][20] = 1; W_lad[5][20] = -1; W_lad[3][20] = -1
    
    # 22. (a10+a11-a20-a21)(b10+b11-b20-b21)
    U_lad[3][21] = 1; U_lad[4][21] = 1; U_lad[6][21] = -1; U_lad[7][21] = -1
    V_lad[3][21] = 1; V_lad[4][21] = 1; V_lad[6][21] = -1; V_lad[7][21] = -1
    W_lad[3][21] = 1; W_lad[4][21] = 1; W_lad[6][21] = -1; W_lad[7][21] = -1
    
    # 23. (a11+a12-a21-a22)(b11+b12-b21-b22)
    U_lad[4][22] = 1; U_lad[5][22] = 1; U_lad[7][22] = -1; U_lad[8][22] = -1
    V_lad[4][22] = 1; V_lad[5][22] = 1; V_lad[7][22] = -1; V_lad[8][22] = -1
    W_lad[4][22] = 1; W_lad[5][22] = 1; W_lad[7][22] = -1; W_lad[8][22] = -1
    
    # Construct full 3x3x6 decomposition
    rank = 46
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for l in range(23):
        # First 3x3x3
        for i in range(9):
            U[i][l] = U_lad[i][l]
            V[i][l] = V_lad[i][l]
            W[i][l] = W_lad[i][l]
        # Second 3x3x3
        for i in range(9):
            U[i][l+23] = U_lad[i][l]
            V[i+9][l+23] = V_lad[i][l]
            W[i+9][l+23] = W_lad[i][l]
            
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    U_school, V_school, W_school, rank_school = make_schoolbook(n, m, p)
    return {"U": U_school, "V": V_school, "W": W_school}
