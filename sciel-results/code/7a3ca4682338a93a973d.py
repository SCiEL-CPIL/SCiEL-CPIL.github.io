import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank)
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x6)
    # A1 is A[:, 0:2]
    # B1 is B[0:2, :]
    # Split B1 into 3 blocks of 2x2: B11, B12, B13
    for j in range(3):
        # B1j is B[0:2, 2j:2j+2]
        # e_j = B[0, 2j], f_j = B[0, 2j+1]
        # g_j = B[1, 2j], h_j = B[1, 2j+1]
        
        # A1 entries:
        # a = A[0,0] -> idx 0
        # b = A[0,1] -> idx 1
        # c = A[1,0] -> idx 5
        # d = A[1,1] -> idx 6
        
        # 1. M1 = (a+b)(e_j+g_j)
        U[0, col] = 1; U[1, col] = 1
        V[2*j, col] = 1; V[6+2*j, col] = 1
        # P11 = C[0, 2j] -> idx 4j
        # P22 = C[1, 2j+1] -> idx 4j+3
        W[4*j, col] = 1
        W[4*j+3, col] = -1
        col += 1
        
        # 2. M2 = d(e_j+g_j)
        U[6, col] = 1
        V[2*j, col] = 1; V[6+2*j, col] = 1
        # P11 (coeff -1), P21 (coeff 1)
        # P21 = C[1, 2j] -> idx 4j+2
        W[4*j, col] = -1
        W[4*j+2, col] = 1
        col += 1
        
        # 3. M3 = (c+d)h_j
        U[5, col] = 1; U[6, col] = 1
        V[7+2*j, col] = 1
        # P12 (coeff -1), P21 (coeff 1)
        # P12 = C[0, 2j+1] -> idx 4j+1
        W[4*j+1, col] = -1
        W[4*j+2, col] = 1
        col += 1
        
        # 4. M4 = a(f_j-h_j)
        U[0, col] = 1
        V[2*j+1, col] = 1; V[7+2*j, col] = -1
        # P22 (coeff -1)
        W[4*j+3, col] = -1
        col += 1
        
        # 5. M5 = (a+b)h_j
        U[0, col] = 1; U[1, col] = 1
        V[7+2*j, col] = 1
        # P12 (coeff 1)
        W[4*j+1, col] = 1
        col += 1
        
        # 6. M6 = (c-d)(e_j+g_j)
        U[5, col] = 1; U[6, col] = -1
        V[2*j, col] = 1; V[6+2*j, col] = 1
        # P11 (coeff 1), P22 (coeff -1)
        W[4*j, col] = 1
        W[4*j+3, col] = -1
        col += 1
        
        # 7. M7 = (c+d)f_j
        U[5, col] = 1; U[6, col] = 1
        V[2*j+1, col] = 1
        # P12 (coeff 1)
        W[4*j+1, col] = 1
        col += 1

    # Part 2: A2 (2x3) * B2 (3x6)
    # A2 is A[:, 2:5]
    # A2a is A[:, 2:4] (2x2)
    # A2b is A[:, 4] (2x1)
    # B2 is B[2:5, :]
    # Split B2 into 3 blocks of 3x2: B21, B22, B23
    for j in range(3):
        # B2j is B[2:5, 2j:2j+2]
        # B2ja is B[2:4, 2j:2j+2] (2x2)
        # B2jb is B[4, 2j:2j+2] (1x2)
        
        # --- Subpart 2a: A2a * B2ja (2x2x2) ---
        # A2a entries:
        # a' = A[0,2] -> idx 2
        # b' = A[0,3] -> idx 3
        # c' = A[1,2] -> idx 7
        # d' = A[1,3] -> idx 8
        
        # B2ja entries:
        # e' = B[2, 2j] -> idx 12+2j
        # f' = B[2, 2j+1] -> idx 13+2j
        # g' = B[3, 2j] -> idx 18+2j
        # h' = B[3, 2j+1] -> idx 19+2j
        
        # 1. M1' = (a'+b')(e'+g')
        U[2, col] = 1; U[3, col] = 1
        V[12+2*j, col] = 1; V[18+2*j, col] = 1
        W[4*j, col] = 1
        W[4*j+3, col] = -1
        col += 1
        
        # 2. M2' = d'(e'+g')
        U[8, col] = 1
        V[12+2*j, col] = 1; V[18+2*j, col] = 1
        W[4*j, col] = -1
        W[4*j+2, col] = 1
        col += 1
        
        # 3. M3' = (c'+d')h'
        U[7, col] = 1; U[8, col] = 1
        V[19+2*j, col] = 1
        W[4*j+1, col] = -1
        W[4*j+2, col] = 1
        col += 1
        
        # 4. M4' = a'(f'-h')
        U[2, col] = 1
        V[13+2*j, col] = 1; V[19+2*j, col] = -1
        W[4*j+3, col] = -1
        col += 1
        
        # 5. M5' = (a'+b')h'
        U[2, col] = 1; U[3, col] = 1
        V[19+2*j, col] = 1
        W[4*j+1, col] = 1
        col += 1
        
        # 6. M6' = (c'-d')(e'+g')
        U[7, col] = 1; U[8, col] = -1
        V[12+2*j, col] = 1; V[18+2*j, col] = 1
        W[4*j, col] = 1
        W[4*j+3, col] = -1
        col += 1
        
        # 7. M7' = (c'+d')f'
        U[7, col] = 1; U[8, col] = 1
        V[13+2*j, col] = 1
        W[4*j+1, col] = 1
        col += 1
        
        # --- Subpart 2b: A2b * B2jb (2x1x2) ---
        # A2b entries:
        # a'' = A[0,4] -> idx 4
        # b'' = A[1,4] -> idx 9
        
        # B2jb entries:
        # x = B[4, 2j] -> idx 24+2j
        # y = B[4, 2j+1] -> idx 25+2j
        
        # 1. a'' * x
        U[4, col] = 1
        V[24+2*j, col] = 1
        W[4*j, col] = 1
        col += 1
        
        # 2. a'' * y
        U[4, col] = 1
        V[25+2*j, col] = 1
        W[4*j+1, col] = 1
        col += 1
        
        # 3. b'' * x
        U[9, col] = 1
        V[24+2*j, col] = 1
        W[4*j+2, col] = 1
        col += 1
        
        # 4. b'' * y
        U[9, col] = 1
        V[25+2*j, col] = 1
        W[4*j+3, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
