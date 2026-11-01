import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # We aim for a rank-28 decomposition (known optimal for 2x3x5).
    # Strategy:
    # Block 1: j=0 (2x1x5) -> rank 10 (schoolbook)
    # Block 2a: j=1,2; k=0,1 (2x2x2) -> rank 7 (Strassen)
    # Block 2b: j=1,2; k=2 (2x2x1) -> rank 4 (schoolbook)
    # Block 2c: j=1,2; k=3,4 (2x2x2) -> rank 7 (Strassen)
    # Total: 10 + 7 + 4 + 7 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, k=0..4
    # Terms: A_{i,0} * B_{0,k} -> C_{i,k}
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    # Local indices: i in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k')
    # Strassen 7 terms for 2x2x2:
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    # M7: (A01-A11)(B10+B11) -> C00 += M7
    
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[1 * n + 0][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[0 * n + 1][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (Schoolbook 2x2x1)
    # Terms: A_{i,j} * B_{j,2} -> C_{i,2}
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    # Local indices: i in {0,1}, j' in {0,1} (j=j'+1), k' in {0,1} (k=k'+3)
    
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

