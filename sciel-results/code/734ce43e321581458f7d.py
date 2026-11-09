import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Construct a rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    The decomposition uses a block-structured approach:
      - A2 (row 2 of A) * B2 (col 2 of B): 2x1x5 block, rank 10
      - A1 (rows 0,1 of A) * B1a (cols 0,1 of B, rows 0,1): 2x2x2 block, Strassen rank 7
      - A1 (rows 0,1 of A) * B1b (cols 0,1 of B, rows 2,3): 2x2x2 block, Strassen rank 7
      - A1 (rows 0,1 of A) * B1c (cols 0,1 of B, row 4): 2x2x1 block, schoolbook rank 4
    Total rank = 10 + 7 + 7 + 4 = 28
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Part 1: A2 * B2 -> C2
    # A indices: i in {0,1}, j=2 => u_idx = i*3 + 2
    # B indices: j=2, k in {0..4} => v_idx = 2*5 + k
    # C indices: k in {0..4}, i in {0,1} => w_idx = k*2 + i
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Part 2: A1 * B1a (2x2x2, Strassen)
    # A indices: i in {0,1}, j in {0,1} => u_idx = i*3 + j
    # B indices: j in {0,1}, k in {0,1} => v_idx = j*5 + k
    # C indices: k in {0,1}, i in {0,1} => w_idx = k*2 + i
    
    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3: A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4: A11(B10-B00)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[0][col] = 1
    col += 1
    
    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[2][col] = 1; W[0][col] = -1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    # Part 3: A1 * B1b (2x2x2, Strassen, k offset by 2)
    # A indices: i in {0,1}, j in {0,1} => u_idx = i*3 + j
    # B indices: j in {0,1}, k in {2,3} => v_idx = j*5 + (2+k_local)
    # C indices: k in {2,3}, i in {0,1} => w_idx = k*2 + i
    
    # M1: (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # M2: (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3: A00(B03-B13)
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4: A11(B12-B02)
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[4][col] = 1
    col += 1
    
    # M5: (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[6][col] = 1; W[4][col] = -1
    col += 1
    
    # M6: (A10-A00)(B02+B03)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7: (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1
    
    # Part 4: A1 * B1c (2x2x1, schoolbook)
    # A indices: i in {0,1}, j in {0,1} => u_idx = i*3 + j
    # B indices: j in {0,1}, k=4 => v_idx = j*5 + 4
    # C indices: k=4, i in {0,1} => w_idx = 4*2 + i
    
    # C04 = A00*B04 + A01*B14
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # C14 = A10*B04 + A11*B14
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
