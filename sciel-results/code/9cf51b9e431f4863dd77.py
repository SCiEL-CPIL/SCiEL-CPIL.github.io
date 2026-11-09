import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = [[0]*7 for _ in range(4)]
    V1 = [[0]*7 for _ in range(4)]
    W1 = [[0]*7 for _ in range(4)]
    
    # c=0
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    # c=1
    U1[1][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1 # W[k,i] -> k=1, i=2 (index 1*2+2=4? No, W1 is 4x7, index k*2+i)
    # W1 index: k*2+i.
    # C21 means i=2 (index 1 in 0-1), k=1 (index 0 in 0-1? No, k is row of B, col of A? No.
    # C[i,k] = sum A[i,j] B[j,k].
    # W[k,i].
    # C21: i=1 (0-indexed), k=0 (0-indexed)? No.
    # i=0,1. j=0,1. k=0,1.
    # C[1,0] (21st element) corresponds to i=1, k=0.
    # W index = k*n + i = 0*2 + 1 = 1.
    # So W1[1][1] = 1.
    
    # c=2
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1 # k=1, i=0 -> 1*2+0=2. C[0,1] (12th)
    W1[3][2] = 1 # k=1, i=1 -> 1*2+1=3. C[1,1] (22nd)
    
    # c=3
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1 # k=0, i=0 -> 0. C[0,0]
    W1[1][3] = 1 # k=0, i=1 -> 1. C[1,0]
    
    # c=4
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1 # k=0, i=0 -> 0. C[0,0]
    W1[2][4] = 1  # k=1, i=0 -> 2. C[0,1]
    
    # c=5
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1  # k=1, i=1 -> 3. C[1,1]
    
    # c=6
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1  # k=0, i=0 -> 0. C[0,0]
    W1[3][6] = 0  # k=1, i=1 -> 3. C[1,1] is 0 for M7?
    # C22 = M1 - M2 + M3 + M6. No M7.
    # So W1[3][6] should be 0.
    
    # Factors for 1x2x3
    U2 = [[0]*6 for _ in range(2)]
    V2 = [[0]*6 for _ in range(6)]
    W2 = [[0]*6 for _ in range(3)]
    
    for c2 in range(6):
        j2 = c2 // 3
        k2 = c2 % 3
        U2[j2][c2] = 1
        V2[j2*3 + k2][c2] = 1
        W2[k2][c2] = 1
        
    # Combine
    for u in range(n*m):
        i = u // m
        j = u % m
        i1 = i
        i2 = 0
        j1 = j // 2
        j2 = j % 2
        u1 = i1 * 2 + j1
        u2 = j2
        for c in range(rank):
            c1 = c // 6
            c2 = c % 6
            U[u][c] = U1[u1][c1] * U2[u2][c2]
            
    for v in range(m*p):
        j = v // p
        k = v % p
        j1 = j // 2
        j2 = j % 2
        k1 = k // 3
        k2 = k % 3
        v1 = j1 * 2 + k1
        v2 = j2 * 3 + k2
        for c in range(rank):
            c1 = c // 6
            c2 = c % 6
            V[v][c] = V1[v1][c1] * V2[v2][c2]
            
    for w in range(p*n):
        k = w // n
        i = w % n
        k1 = k // 3
        k2 = k % 3
        i1 = i
        i2 = 0
        w1 = k1 * 2 + i1
        w2 = k2
        for c in range(rank):
            c1 = c // 6
            c2 = c % 6
            W[w][c] = W1[w1][c1] * W2[w2][c2]
            
    return {"U": U, "V": V, "W": W}
