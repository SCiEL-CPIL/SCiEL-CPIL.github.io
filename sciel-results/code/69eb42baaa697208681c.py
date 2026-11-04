import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Rank = 28 (Strassen) + 16 (A22) + 16 (A2j) + 8 (A22_B2k) = 68
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for i in {0,1}, j in {0,1}
    # 4 blocks of k: [0,1], [2,3], [4,5], [6,7]
    for b in range(4):
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # m1 = (a+d)(e+h)
        # U: (0,0)=1, (1,1)=1
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        # V: (0,0)=1, (1,1)=1 -> indices (0, k1), (1, k2) NO.
        # V is for B. B is 3x8.
        # In the 2x2 block of B (rows 0,1; cols k1,k2):
        # e = B[0, k1], f = B[0, k2]
        # g = B[1, k1], h = B[1, k2]
        # So e+h means V[0*8+k1]=1, V[1*8+k2]=1?
        # Wait. m1 = (a+d)(e+h).
        # e is B[0,0] of the block, h is B[1,1] of the block.
        # So V[0*8+k1] = 1, V[1*8+k2] = 1.
        V[0*8+k1][col] = 1
        V[1*8+k2][col] = 1
        # W: C[0,0] coeff 1, C[1,1] coeff 1
        # C[0,0] -> i=0, k=k1. W[k1*3+0] = 1
        # C[1,1] -> i=1, k=k2. W[k2*3+1] = 1
        W[k1*3+0][col] = 1
        W[k2*3+1][col] = 1
        col += 1
        
        # m2 = (c+d)e
        # U: (1,0)=1, (1,1)=1
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        # V: e = B[0, k1]. V[0*8+k1] = 1
        V[0*8+k1][col] = 1
        # W: C[1,0] coeff 1, C[1,1] coeff -1
        # C[1,0] -> i=1, k=k1. W[k1*3+1] = 1
        # C[1,1] -> i=1, k=k2. W[k2*3+1] = -1
        W[k1*3+1][col] = 1
        W[k2*3+1][col] = -1
        col += 1
        
        # m3 = a(f-h)
        # U: (0,0)=1
        U[0*3+0][col] = 1
        # V: f-h = B[0, k2] - B[1, k2]. V[0*8+k2]=1, V[1*8+k2]=-1
        V[0*8+k2][col] = 1
        V[1*8+k2][col] = -1
        # W: C[0,1] coeff 1, C[1,1] coeff 1
        # C[0,1] -> i=0, k=k2. W[k2*3+0] = 1
        # C[1,1] -> i=1, k=k2. W[k2*3+1] = 1
        W[k2*3+0][col] = 1
        W[k2*3+1][col] = 1
        col += 1
        
        # m4 = d(g-e)
        # U: (1,1)=1
        U[1*3+1][col] = 1
        # V: g-e = B[1, k1] - B[0, k1]. V[1*8+k1]=1, V[0*8+k1]=-1
        V[1*8+k1][col] = 1
        V[0*8+k1][col] = -1
        # W: C[0,0] coeff -1, C[0,1] coeff 1
        # C[0,0] -> i=0, k=k1. W[k1*3+0] = -1
        # C[0,1] -> i=0, k=k2. W[k2*3+0] = 1
        W[k1*3+0][col] = -1
        W[k2*3+0][col] = 1
        col += 1
        
        # m5 = (a+b)h
        # U: (0,0)=1, (0,1)=1
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        # V: h = B[1, k2]. V[1*8+k2]=1
        V[1*8+k2][col] = 1
        # W: C[0,0] coeff 1, C[1,0] coeff 1
        # C[0,0] -> i=0, k=k1. W[k1*3+0] = 1
        # C[1,0] -> i=1, k=k1. W[k1*3+1] = 1
        W[k1*3+0][col] = 1
        W[k1*3+1][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        # U: (1,0)=1, (0,0)=-1
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        # V: e+f = B[0, k1] + B[0, k2]. V[0*8+k1]=1, V[0*8+k2]=1
        V[0*8+k1][col] = 1
        V[0*8+k2][col] = 1
        # W: C[1,1] coeff 1
        # C[1,1] -> i=1, k=k2. W[k2*3+1] = 1
        W[k2*3+1][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        # U: (0,1)=1, (1,1)=-1
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        # V: g+h = B[1, k1] + B[1, k2]. V[1*8+k1]=1, V[1*8+k2]=1
        V[1*8+k1][col] = 1
        V[1*8+k2][col] = 1
        # W: C[0,0] coeff -1
        # C[0,0] -> i=0, k=k1. W[k1*3+0] = -1
        W[k1*3+0][col] = -1
        col += 1

    # Schoolbook for i in {0,1}, j=2
    # A[i,2] * B[2,k]
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*3+i][col] = 1
            col += 1

    # Schoolbook for i=2, j in {0,1}
    # A[2,j] * B[j,k]
    for j in range(2):
        for k in range(8):
            U[2*3+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1

    # Schoolbook for i=2, j=2
    # A[2,2] * B[2,k]
    for k in range(8):
        U[2*3+2][col] = 1
        V[2*8+k][col] = 1
        W[k*3+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
