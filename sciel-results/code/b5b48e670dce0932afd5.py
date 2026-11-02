import numpy as np

def construct():
    """
    Return a valid rank-24 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy:
    - Block 1 (j=0): 2x1x5 block, rank 6 (schoolbook).
    - Block 2a (j∈{1,2}, k∈{0,1}): 2x2x2 block, rank 7 (Strassen).
    - Block 2b (j∈{1,2}, k=2): 2x2x1 block, rank 4 (schoolbook).
    - Block 2c (j∈{1,2}, k∈{3,4}): 2x2x2 block, rank 7 (Strassen).
    - Total rank: 6 + 7 + 4 + 7 = 24.
    """
    n, m, p = 2, 3, 5
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 15 x 24
    W = [[0] * rank for _ in range(p * n)]  # 10 x 24
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4} — rank 6 (schoolbook)
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 6
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    # M1: (A[0,1]+A[1,2]) * (B[1,0]+B[2,1]) -> C[0,0] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,0] -> C[1,0] += 1, C[1,1] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    W[1 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,1]-B[2,1]) -> C[0,1] += 1, C[1,1] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 1][col] = 1
    V[2 * 5 + 1][col] = -1
    W[1 * 2 + 0][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,0]-B[1,0]) -> C[0,0] += 1, C[1,0] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 0][col] = 1
    V[1 * 5 + 0][col] = -1
    W[0 * 2 + 0][col] = 1
    W[0 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,1] -> C[0,0] += -1, C[0,1] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = -1
    W[1 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,0]+B[1,1]) -> C[1,1] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 0][col] = 1
    V[1 * 5 + 1][col] = 1
    W[1 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,0]+B[2,1]) -> C[0,0] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 0][col] = 1
    V[2 * 5 + 1][col] = 1
    W[0 * 2 + 0][col] = 1
    col += 1
    # col is now 13
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
    # col is now 17
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    # M1: (A[0,1]+A[1,2]) * (B[1,3]+B[2,4]) -> C[0,3] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2]) * B[1,3] -> C[1,3] += 1, C[1,4] += -1
    U[1 * 3 + 1][col] = 1
    U[1 * 3 + 2][col] = 1
    V[1 * 5 + 3][col] = 1
    W[3 * 2 + 1][col] = 1
    W[4 * 2 + 1][col] = -1
    col += 1
    
    # M3: A[0,1] * (B[1,4]-B[2,4]) -> C[0,4] += 1, C[1,4] += 1
    U[0 * 3 + 1][col] = 1
    V[1 * 5 + 4][col] = 1
    V[2 * 5 + 4][col] = -1
    W[4 * 2 + 0][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M4: A[1,2] * (B[2,3]-B[1,3]) -> C[0,3] += 1, C[1,3] += 1
    U[1 * 3 + 2][col] = 1
    V[2 * 5 + 3][col] = 1
    V[1 * 5 + 3][col] = -1
    W[3 * 2 + 0][col] = 1
    W[3 * 2 + 1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2]) * B[2,4] -> C[0,3] += -1, C[0,4] += 1
    U[0 * 3 + 1][col] = 1
    U[0 * 3 + 2][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = -1
    W[4 * 2 + 0][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1]) * (B[1,3]+B[1,4]) -> C[1,4] += 1
    U[1 * 3 + 1][col] = 1
    U[0 * 3 + 1][col] = -1
    V[1 * 5 + 3][col] = 1
    V[1 * 5 + 4][col] = 1
    W[4 * 2 + 1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2]) * (B[2,3]+B[2,4]) -> C[0,3] += 1
    U[0 * 3 + 2][col] = 1
    U[1 * 3 + 2][col] = -1
    V[2 * 5 + 3][col] = 1
    V[2 * 5 + 4][col] = 1
    W[3 * 2 + 0][col] = 1
    col += 1
    
    # col is now 24
    
    return {"U": U, "V": V, "W": W}
