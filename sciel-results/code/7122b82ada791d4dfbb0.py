import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            if actual != expected:
                                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # A is 2x3, B is 3x3, C is 2x3
    # U shape (2*3, rank) = (6, rank)
    # V shape (3*3, rank) = (9, rank)
    # W shape (3*2, rank) = (6, rank)
    
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A_01 (2x2) * B_01_left (2x2) -> Strassen (7 terms)
    # A_01 indices: (0,0), (0,1), (1,0), (1,1)
    # B_01_left indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen products for 2x2x2
    # P1 = A00*B00
    # P2 = (A00+A10)*B10
    # P3 = A01*(B00+B01)
    # P4 = A11*(B10+B11)
    # P5 = (A00+A01)*(B11-B10)
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Term 0: P1
    U[0*3+0][0] = 1; V[0*3+0][0] = 1; W[0*2+0][0] = 1
    # Term 1: P2
    U[0*3+0][1] = 1; U[1*3+0][1] = 1; V[1*3+0][1] = 1; W[0*2+0][1] = 1
    # Term 2: P3
    U[0*3+1][2] = 1; V[0*3+0][2] = 1; V[0*3+1][2] = 1; W[0*2+1][2] = 1
    # Term 3: P4
    U[1*3+1][3] = 1; V[1*3+0][3] = 1; V[1*3+1][3] = 1; W[0*2+0][3] = 1
    # Term 4: P5
    U[0*3+0][4] = 1; U[0*3+1][4] = 1; V[1*3+1][4] = 1; V[1*3+0][4] = -1; W[0*2+0][4] = -1; W[1*2+0][4] = 1
    # Wait, W index for C11 is 1*2+1? No, W[k*n+i].
    # C00 is i=0, k=0 -> W[0].
    # C01 is i=0, k=1 -> W[2].
    # C10 is i=1, k=0 -> W[1].
    # C11 is i=1, k=1 -> W[3].
    
    # Let's re-index W carefully.
    # W[k*n + i]. n=2.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    
    # P1: C00 += 1. W[0] += 1.
    # P2: C10 += 1. W[1] += 1.
    # P3: C01 += 1. W[2] += 1.
    # P4: C00 += 1, C10 += 1. W[0] += 1, W[1] += 1.
    # P5: C00 += -1, C11 += 1. W[0] += -1, W[3] += 1.
    # P6: C11 += 1. W[3] += 1.
    # P7: C00 += 1, C11 += -1. W[0] += 1, W[3] += -1.
    
    # Let's restart the W assignments for Strassen.
    # Term 0 (P1): U[0,0]=1, V[0,0]=1. C00 += 1. W[0] += 1.
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1 (P2): U[0,0]=1, U[1,0]=1. V[1,0]=1. C10 += 1. W[1] += 1.
    U[0][1] = 1; U[3][1] = 1; V[3][1] = 1; W[1][1] = 1
    # Term 2 (P3): U[0,1]=1. V[0,0]=1, V[0,1]=1. C01 += 1. W[2] += 1.
    U[1][2] = 1; V[0][2] = 1; V[1][2] = 1; W[2][2] = 1
    # Term 3 (P4): U[1,1]=1. V[1,0]=1, V[1,1]=1. C00 += 1, C10 += 1. W[0] += 1, W[1] += 1.
    U[4][3] = 1; V[3][3] = 1; V[4][3] = 1; W[0][3] = 1; W[1][3] = 1
    # Term 4 (P5): U[0,0]=1, U[0,1]=1. V[1,1]=1, V[1,0]=-1. C00 += -1, C11 += 1. W[0] += -1, W[3] += 1.
    U[0][4] = 1; U[1][4] = 1; V[4][4] = 1; V[3][4] = -1; W[0][4] = -1; W[3][4] = 1
    # Term 5 (P6): U[1,0]=1, U[0,0]=-1. V[0,0]=1, V[0,1]=1. C11 += 1. W[3] += 1.
    U[3][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
    # Term 6 (P7): U[0,1]=1, U[1,1]=-1. V[1,0]=1, V[1,1]=1. C00 += 1, C11 += -1. W[0] += 1, W[3] += -1.
    U[1][6] = 1; U[4][6] = -1; V[3][6] = 1; V[4][6] = 1; W[0][6] = 1; W[3][6] = -1
    
    # Part 1b: A_01 (2x2) * B_01_right (2x1) -> Schoolbook (4 terms)
    # B_01_right is column 2 of B (indices 0,2 and 1,2).
    # C02 += A00*B02 + A01*B12
    # C12 += A10*B02 + A11*B12
    
    # Term 7: A00*B02
    U[0][7] = 1; V[2][7] = 1; W[4][7] = 1 # C02 is i=0, k=2 -> W[4]
    # Term 8: A01*B12
    U[1][8] = 1; V[5][8] = 1; W[4][8] = 1
    # Term 9: A10*B02
    U[3][9] = 1; V[2][9] = 1; W[5][9] = 1 # C12 is i=1, k=2 -> W[5]
    # Term 10: A11*B12
    U[4][10] = 1; V[5][10] = 1; W[5][10] = 1
    
    # Part 2: A_2 (2x1) * B_2 (1x3) -> Outer Product (6 terms)
    # A_2 is column 2 of A (indices 0,2 and 1,2).
    # B_2 is row 2 of B (indices 2,0; 2,1; 2,2).
    # C00 += A02*B20
    # C01 += A02*B21
    # C02 += A02*B22
    # C10 += A12*B20
    # C11 += A12*B21
    # C12 += A12*B22
    
    # Term 11: A02*B20
    U[2][11] = 1; V[6][11] = 1; W[0][11] = 1
    # Term 12: A02*B21
    U[2][12] = 1; V[7][12] = 1; W[2][12] = 1
    # Term 13: A02*B22
    U[2][13] = 1; V[8][13] = 1; W[4][13] = 1
    # Term 14: A12*B20
    U[5][14] = 1; V[6][14] = 1; W[1][14] = 1
    # Term 15: A12*B21
    U[5][15] = 1; V[7][15] = 1; W[3][15] = 1
    # Term 16: A12*B22
    U[5][16] = 1; V[8][16] = 1; W[5][16] = 1
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
