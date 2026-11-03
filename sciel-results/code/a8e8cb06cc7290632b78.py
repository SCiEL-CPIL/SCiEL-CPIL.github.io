import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 78
    # 50 for 2x4x7 part
    # 28 for 1x4x7 part
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    # Helper to add Strassen 2x2x2 block
    def add_strassen(i_base, j_base, k_base):
        # i_base, j_base, k_base are the starting global indices
        # Local indices a=0,1; b=0,1; c=0,1
        # Global i = i_base + a
        # Global j = j_base + b
        # Global k = k_base + c
        
        # u, v, alpha coefficients for 7 Strassen products
        # P1 = A00(B00+B11)
        # P2 = A10(B00+B01)
        # P3 = A01(B10+B11)
        # P4 = (A00-A10)(B11-B01)
        # P5 = A11(B00+B10)
        # P6 = (A01-A11)(B01-B00)
        # P7 = (A10-A00)(B10-B11)
        
        # u[a, b, t]
        u = [
            [0]*7, [0]*7, # a=0
            [0]*7, [0]*7  # a=1
        ]
        # v[b, c, t]
        v = [
            [0]*7, [0]*7, # b=0
            [0]*7, [0]*7  # b=1
        ]
        # alpha[a, c, t] (contribution to C_ac)
        alpha = [
            [0]*7, [0]*7, # a=0
            [0]*7, [0]*7  # a=1
        ]
        
        # P1
        u[0][0][0] = 1
        v[0][0][0] = 1
        v[1][1][0] = 1
        alpha[0][0][0] = 1
        
        # P2
        u[1][0][1] = 1
        v[0][0][1] = 1
        v[0][1][1] = 1
        alpha[1][0][1] = 1
        alpha[1][1][1] = 1
        
        # P3
        u[0][1][2] = 1
        v[1][0][2] = 1
        v[1][1][2] = 1
        alpha[0][1][2] = 1
        alpha[1][1][2] = 1
        
        # P4
        u[0][0][3] = 1
        u[1][0][3] = -1
        v[1][1][3] = 1
        v[0][1][3] = -1
        alpha[0][0][3] = 1
        alpha[0][1][3] = -1
        
        # P5
        u[1][1][4] = 1
        v[0][0][4] = 1
        v[1][0][4] = 1
        alpha[0][0][4] = -1
        alpha[1][1][4] = 1
        
        # P6
        u[0][1][5] = 1
        u[1][1][5] = -1
        v[0][1][5] = 1
        v[0][0][5] = -1
        alpha[0][1][5] = -1
        alpha[1][0][5] = 1
        
        # P7
        u[1][0][6] = 1
        u[0][0][6] = -1
        v[1][0][6] = 1
        v[1][1][6] = -1
        alpha[0][0][6] = 1
        alpha[1][0][6] = -1

        for t in range(7):
            for a in range(2):
                for b in range(2):
                    if u[a][b][t] != 0:
                        i = i_base + a
                        j = j_base + b
                        U[i * m + j][col] += u[a][b][t]
            
            for b in range(2):
                for c in range(2):
                    if v[b][c][t] != 0:
                        j = j_base + b
                        k = k_base + c
                        V[j * p + k][col] += v[b][c][t]
                        
            for a in range(2):
                for c in range(2):
                    if alpha[a][c][t] != 0:
                        i = i_base + a
                        k = k_base + c
                        W[k * n + i][col] += alpha[a][c][t]
                        
            col += 1

    # Block 1: i in 0,1 (2 rows)
    # Split j into 0,1 and 2,3
    # Split k into 0,1; 2,3; 4,5; 6
    
    # j=0,1
    add_strassen(0, 0, 0) # k=0,1
    add_strassen(0, 0, 2) # k=2,3
    add_strassen(0, 0, 4) # k=4,5
    # k=6 (2x2x1)
    add_schoolbook(0, 0, 6)
    add_schoolbook(0, 1, 6)
    add_schoolbook(1, 0, 6)
    add_schoolbook(1, 1, 6)
    
    # j=2,3
    add_strassen(0, 2, 0) # k=0,1
    add_strassen(0, 2, 2) # k=2,3
    add_strassen(0, 2, 4) # k=4,5
    # k=6 (2x2x1)
    add_schoolbook(0, 2, 6)
    add_schoolbook(0, 3, 6)
    add_schoolbook(1, 2, 6)
    add_schoolbook(1, 3, 6)
    
    # Block 2: i in 2 (1 row)
    # Schoolbook for all j, k
    for j in range(4):
        for k in range(7):
            add_schoolbook(2, j, k)
            
    return {"U": U, "V": V, "W": W}
