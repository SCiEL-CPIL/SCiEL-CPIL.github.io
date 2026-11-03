import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    total_rank_school = n * m * p
    U_school = np.zeros((n * m, total_rank_school))
    V_school = np.zeros((m * p, total_rank_school))
    W_school = np.zeros((p * n, total_rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Rank-26 decomposition for 3x3x3 (block Strassen)
    # Rank-36 decomposition for 3x3x4 (schoolbook)
    # Total rank: 62
    
    rank_333 = 26
    rank_334 = 36
    total_rank = rank_333 + rank_334
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # 3x3x3 part: columns 0-2 of B
    # Strassen's 7 products for A11*B11 (columns 0-6)
    # A11 = A[0:2, 0:2], B11 = B[0:2, 0:2]
    
    # M1 = (A11[0,0]+A11[1,1])*(B11[0,0]+B11[1,1])
    # u: A[0,0] and A[1,1] -> indices 0*3+0=0, 1*3+1=4
    # v: B[0,0] and B[1,1] -> indices 0*3+0=0, 1*3+1=4
    # w: contributes to C[0,0] and C[1,1]
    # C[0,0] = M1 + ... -> W index 0*3+0=0
    # C[1,1] = M1 - ... -> W index 1*3+1=4
    
    col = 0
    # M1
    U[0, col] = 1; U[4, col] = 1  # A[0,0], A[1,1]
    V[0, col] = 1; V[4, col] = 1  # B[0,0], B[1,1]
    W[0, col] = 1; W[4, col] = 1  # C[0,0], C[1,1]
    col += 1
    
    # M2 = (A11[1,0]+A11[1,1])*B11[0,0]
    # A[1,0]=3, A[1,1]=4; B[0,0]=0
    # C[1,0]=3, C[1,1]=4
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # M3 = A11[0,0]*(B11[0,1]-B11[1,1])
    # A[0,0]=0; B[0,1]=1, B[1,1]=4
    # C[0,1]=1, C[1,1]=4
    U[0, col] = 1
    V[1, col] = 1; V[4, col] = -1
    W[1, col] = 1; W[4, col] = 1
    col += 1
    
    # M4 = A11[1,1]*(B11[1,0]-B11[0,0])
    # A[1,1]=4; B[1,0]=3, B[0,0]=0
    # C[0,0]=0, C[1,0]=3
    U[4, col] = 1
    V[3, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # M5 = (A11[0,0]+A11[0,1])*B11[1,1]
    # A[0,0]=0, A[0,1]=1; B[1,1]=4
    # C[0,1]=1, C[1,1]=4
    U[0, col] = 1; U[1, col] = 1
    V[4, col] = 1
    W[1, col] = 1; W[4, col] = 1
    col += 1
    
    # M6 = (A11[1,0]-A11[0,0])*(B11[0,0]+B11[0,1])
    # A[1,0]=3, A[0,0]=0; B[0,0]=0, B[0,1]=1
    # C[1,0]=3, C[1,1]=4
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # M7 = (A11[0,1]-A11[1,1])*(B11[1,0]+B11[1,1])
    # A[0,1]=1, A[1,1]=4; B[1,0]=3, B[1,1]=4
    # C[0,0]=0, C[0,1]=1
    U[1, col] = 1; U[4, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # Now handle the remaining terms
    # a12*b21: A[0:2,2] * B[2,0:2] -> C[0:2,0:2]
    # A[0,2]=2, A[1,2]=5; B[2,0]=6, B[2,1]=7
    # C[0,0]=0, C[0,1]=1, C[1,0]=3, C[1,1]=4
    
    # Term: A[0,2]*B[2,0] -> C[0,0]
    U[2, col] = 1; V[6, col] = 1; W[0, col] = 1; col += 1
    # Term: A[0,2]*B[2,1] -> C[0,1]
    U[2, col] = 1; V[7, col] = 1; W[1, col] = 1; col += 1
    # Term: A[1,2]*B[2,0] -> C[1,0]
    U[5, col] = 1; V[6, col] = 1; W[3, col] = 1; col += 1
    # Term: A[1,2]*B[2,1] -> C[1,1]
    U[5, col] = 1; V[7, col] = 1; W[4, col] = 1; col += 1
    
    # A11*b12: A[0:2,0:2] * B[0:2,2] -> C[0:2,2]
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4; B[0,2]=8, B[1,2]=9
    # C[0,2]=6, C[1,2]=7
    
    # Term: A[0,0]*B[0,2] -> C[0,2]
    U[0, col] = 1; V[8, col] = 1; W[6, col] = 1; col += 1
    # Term: A[0,1]*B[1,2] -> C[0,2]
    U[1, col] = 1; V[9, col] = 1; W[6, col] = 1; col += 1
    # Term: A[1,0]*B[0,2] -> C[1,2]
    U[3, col] = 1; V[8, col] = 1; W[7, col] = 1; col += 1
    # Term: A[1,1]*B[1,2] -> C[1,2]
    U[4, col] = 1; V[9, col] = 1; W[7, col] = 1; col += 1
    
    # a12*b22: A[0:2,2] * B[2,2] -> C[0:2,2]
    # A[0,2]=2, A[1,2]=5; B[2,2]=10
    # C[0,2]=6, C[1,2]=7
    
    # Term: A[0,2]*B[2,2] -> C[0,2]
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    # Term: A[1,2]*B[2,2] -> C[1,2]
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    
    # a21*B11: A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    # A[2,0]=6, A[2,1]=7; B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    # C[2,0]=12, C[2,1]=13
    
    # Term: A[2,0]*B[0,0] -> C[2,0]
    U[6, col] = 1; V[0, col] = 1; W[12, col] = 1; col += 1
    # Term: A[2,0]*B[1,0] -> C[2,0]
    U[6, col] = 1; V[3, col] = 1; W[12, col] = 1; col += 1
    # Term: A[2,1]*B[0,1] -> C[2,1]
    U[7, col] = 1; V[1, col] = 1; W[13, col] = 1; col += 1
    # Term: A[2,1]*B[1,1] -> C[2,1]
    U[7, col] = 1; V[4, col] = 1; W[13, col] = 1; col += 1
    
    # a22*b21: A[2,2] * B[2,0:2] -> C[2,0:2]
    # A[2,2]=8; B[2,0]=6, B[2,1]=7
    # C[2,0]=12, C[2,1]=13
    
    # Term: A[2,2]*B[2,0] -> C[2,0]
    U[8, col] = 1; V[6, col] = 1; W[12, col] = 1; col += 1
    # Term: A[2,2]*B[2,1] -> C[2,1]
    U[8, col] = 1; V[7, col] = 1; W[13, col] = 1; col += 1
    
    # a21*b12: A[2,0:2] * B[0:2,2] -> C[2,2]
    # A[2,0]=6, A[2,1]=7; B[0,2]=8, B[1,2]=9
    # C[2,2]=14
    
    # Term: A[2,0]*B[0,2] -> C[2,2]
    U[6, col] = 1; V[8, col] = 1; W[14, col] = 1; col += 1
    # Term: A[2,1]*B[1,2] -> C[2,2]
    U[7, col] = 1; V[9, col] = 1; W[14, col] = 1; col += 1
    
    # a22*b22: A[2,2] * B[2,2] -> C[2,2]
    U[8, col] = 1; V[10, col] = 1; W[14, col] = 1; col += 1
    
    # Verify col == rank_333
    assert col == rank_333, f"Expected {rank_333}, got {col}"
    
    # 3x3x4 part: columns 3-6 of B (schoolbook, rank 36)
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
