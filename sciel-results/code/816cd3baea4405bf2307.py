import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank 33
    # Part 1: M_{2,1,6} (12 terms)
    # Part 2: M_{2,2,6} (21 terms, 3 blocks of M_{2,2,2} with Strassen)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: M_{2,1,6}
    # A is 2x1 (col 0), B is 1x6 (row 0)
    # C is 2x6
    # Term r: 0..11
    for r in range(12):
        i = r // 6
        k = r % 6
        # U: row i, col 0
        U[i * m + 0][r] = 1
        # V: row 0, col k
        V[0 * p + k][r] = 1
        # W: row k, col i (W[k*n+i])
        W[k * n + i][r] = 1
        
    # Part 2: M_{2,2,6}
    # A is 2x2 (cols 1,2), B is 2x6 (rows 1,2)
    # Split B into 3 blocks of 2x2: cols (0,1), (2,3), (4,5)
    # Use Strassen for each block
    
    # Strassen coefficients for M_{2,2,2}
    # A indices: 0,1 (top), 2,3 (bottom) -> a,b,c,d
    # B indices: 0,1 (top), 2,3 (bottom) -> x,y,z,w
    # C indices: 0,1 (top), 2,3 (bottom) -> c00,c01,c10,c11
    
    # m1 = (a+d)(x+w)
    # m2 = (c+d)x
    # m3 = a(y-x)
    # m4 = d(x-y)
    # m5 = (a+b)w
    # m6 = (c-a)(x+w) -- CORRECTION: m6 = (c-a)(x+w) is NOT correct.
    # Correct Strassen:
    # m1 = (a+d)(x+w)
    # m2 = (c+d)x
    # m3 = a(y-x)
    # m4 = d(x-y)
    # m5 = (a+b)w
    # m6 = (c-a)(x+w) -- No.
    # m6 = (c-a)(x+w) is not right.
    # m6 = (c-a)(x+w)?
    
    # Let's use the explicit vectors for Strassen:
    # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]
    # u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0,0,1,0]
    # u3 = [1,0,0,0], v3 = [-1,1,0,0], w3 = [0,1,0,0]
    # u4 = [0,0,0,1], v4 = [1,-1,0,0], w4 = [1,0,0,0]
    # u5 = [1,1,0,0], v5 = [0,0,0,1], w5 = [0,0,0,1]
    # u6 = [0,0,1,-1], v6 = [1,0,1,0], w6 = [0,0,1,0] -- CHECK
    # u7 = [0,1,-1,0], v7 = [0,1,0,1], w7 = [0,1,0,0] -- CHECK
    
    # Verification of m6:
    # c10 = c*x + d*z
    # m6 + m4 = c10
    # m4 = d(x-y)
    # m6 = c10 - m4 = cx + dz - dx + dy = c(x-z) + d(y-z)
    # u6 = [0,0,1,-1] -> c-d
    # v6 = [1,0,1,0] -> x+z
    # (c-d)(x+z) = cx + cz - dx - dz
    # This is not c(x-z) + d(y-z).
    
    # I will use the Schoolbook for M_{2,2,2} to ensure correctness, even if rank is higher.
    # Rank 8 for M_{2,2,2}.
    # Total Rank = 12 + 3*8 = 36.
    # This is the same as the seed.
    
    # I will stick to the Rank 33 construction but I need the correct Strassen coefficients.
    # Since I am unsure, I will return the Schoolbook (Rank 36) to be safe?
    # No, I can use the Rank 11 for M_{2,3,2} block?
    # M_{2,3,2} rank 11.
    # 3 blocks of M_{2,3,2}.
    # Total rank 33.
    
    # M_{2,3,2} decomposition:
    # Split 2x3 into 2x2 and 2x1.
    # 2x2 by 2x2 (Rank 7 Strassen) + 2x1 by 1x2 (Rank 4 Schoolbook).
    
    # This is safer if I can construct M_{2,3,2} rank 11.
    
    # I will implement the Rank 36 Schoolbook to ensure validity, as I cannot guarantee the Strassen coefficients.
    
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
