import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 104 decomposition using Strassen for 2x2x2 subblock
    # and schoolbook for the rest.
    
    # Identify the 2x2x2 block indices
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    # Global indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # We will build U, V, W as lists of lists (integers)
    # Total rank = 104
    # 7 columns for Strassen, 97 columns for Schoolbook
    
    rank = 104
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: Strassen Block (Columns 0 to 6) ---
    
    # P1 = (A00 + A10)(B00 + B10)
    # U: A00(idx 0), A10(idx 5)
    # V: B00(idx 0), B10(idx 7)
    # W: C00(idx 0), C11(idx 4)
    U[0][0] = 1; U[5][0] = 1
    V[0][0] = 1; V[7][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # P2 = (A10 + A11)B00
    # U: A10(idx 5), A11(idx 6)
    # V: B00(idx 0)
    # W: C10(idx 1), -C11(idx 4)
    U[5][1] = 1; U[6][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1
    
    # P3 = A00(B01 - B11)
    # U: A00(idx 0)
    # V: B01(idx 1), -B11(idx 8)
    # W: C01(idx 3), C11(idx 4)
    U[0][2] = 1
    V[1][2] = 1; V[8][2] = -1
    W[3][2] = 1; W[4][2] = 1
    
    # P4 = A11(B10 - B00)
    # U: A11(idx 6)
    # V: B10(idx 7), -B00(idx 0)
    # W: C10(idx 1), C11(idx 4)
    U[6][3] = 1
    V[7][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[4][3] = 1
    
    # P5 = (A00 + A01)B11
    # U: A00(idx 0), A01(idx 1)
    # V: B11(idx 8)
    # W: -C00(idx 0), C01(idx 3)
    U[0][4] = 1; U[1][4] = 1
    V[8][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # P6 = (A10 - A00)(B00 + B01)
    # U: A10(idx 5), -A00(idx 0)
    # V: B00(idx 0), B01(idx 1)
    # W: C11(idx 4)
    U[5][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # P7 = (A01 - A11)(B10 + B11)
    # U: A01(idx 1), -A11(idx 6)
    # V: B10(idx 7), B11(idx 8)
    # W: C00(idx 0)
    U[1][6] = 1; U[6][6] = -1
    V[7][6] = 1; V[8][6] = 1
    W[0][6] = 1
    
    # --- Part 2: Schoolbook for remaining terms (Columns 7 to 103) ---
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    # Verify col is 104
    # assert col == 104
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
