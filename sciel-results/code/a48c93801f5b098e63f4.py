import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 3x3x3 using Strassen block decomposition (rank 26)
    # Part 2: 3x3x4 using schoolbook (rank 36)
    # Total rank 62
    
    rank1 = 26
    rank2 = 3 * 3 * 4
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for A22*B22 (rank 7)
    # A22: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B22: B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4
    # C22: C[0,0]=W[0*3+0]=W[0], C[0,1]=W[1*3+0]=W[3], C[1,0]=W[0*3+1]=W[1], C[1,1]=W[1*3+1]=W[4]
    # Wait, C[k,i] where k is col of B, i is row of A
    # C[0,0]: k=0,i=0 → W[0]
    # C[0,1]: k=0,i=1 → W[1]  
    # C[1,0]: k=1,i=0 → W[3]
    # C[1,1]: k=1,i=1 → W[4]
    
    # P1 = (a11+a22)(b11+b22)
    # C[0,0]+=P1, C[1,1]+=P1, C[0,1]-=P1, C[1,0]+=P1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1; W[1][col] = -1; W[3][col] = 1
    col += 1
    
    # P2 = (a21+a22)b11
    # C[1,0]+=P2, C[1,1]-=P2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # P3 = a11(b21-b22)
    # C[0,1]+=P3, C[1,1]+=P3
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[4][col] = 1
    col += 1
    
    # P4 = a22(b12-b11)
    # C[0,0]+=P4, C[1,0]+=P4
    U[4][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # P5 = (a11+a12)b22
    # C[0,0]-=P5, C[1,1]+=P5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[4][col] = 1
    col += 1
    
    # P6 = (a21-a11)(b11+b12)
    # C[1,1]+=P6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7 = (a12-a22)(b21+b22)
    # C[0,0]+=P7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # a1*b2T (rank 4)
    # a1 = [A[0,2], A[1,2]] = U[2], U[5]
    # b2T = [B[2,0], B[2,1]] = V[6], V[7]
    # C[0,0]+=A[0,2]*B[2,0], C[1,0]+=A[0,2]*B[2,1]
    # C[0,1]+=A[1,2]*B[2,0], C[1,1]+=A[1,2]*B[2,1]
    
    U[2][col] = 1
    V[6][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[7][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[6][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[7][col] = 1
    W[4][col] = 1
    col += 1
    
    # A22*b1 (rank 4)
    # A22: U[0],U[1],U[3],U[4]
    # b1 = [B[0,2], B[1,2]] = V[2], V[5]
    # C[0,2]+=A[0,0]*B[0,2]+A[0,1]*B[1,2]
    # C[1,2]+=A[1,0]*B[0,2]+A[1,1]*B[1,2]
    # W[2*3+0]=W[6] for C[2,0], W[2*3+1]=W[7] for C[2,1]
    
    U[0][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    U[1][col] = 1
    V[5][col] = 1
    W[6][col] = 1
    col += 1
    
    U[3][col] = 1
    V[2][col] = 1
    W[7][col] = 1
    col += 1
    
    U[4][col] = 1
    V[5][col] = 1
    W[7][col] = 1
    col += 1
    
    # a1*b33 (rank 2)
    # a1 = U[2], U[5], b33 = B[2,2] = V[8]
    # C[2,0]+=A[0,2]*B[2,2], C[2,1]+=A[1,2]*B[2,2]
    
    U[2][col] = 1
    V[8][col] = 1
    W[6][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[7][col] = 1
    col += 1
    
    # a2T*B22 (rank 4)
    # a2T = [A[2,0], A[2,1]] = U[6], U[7]
    # B22: V[0],V[1],V[3],V[4]
    # C[2,0]+=A[2,0]*B[0,0]+A[2,1]*B[1,0]
    # C[2,1]+=A[2,0]*B[0,1]+A[2,1]*B[1,1]
    # W[2*3+0]=W[6], W[2*3+1]=W[7]
    
    U[6][col] = 1
    V[0][col] = 1
    W[6][col] = 1
    col += 1
    
    U[7][col] = 1
    V[3][col] = 1
    W[6][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[7][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[7][col] = 1
    col += 1
    
    # a33*b2T (rank 2)
    # a33 = A[2,2] = U[8], b2T = V[6], V[7]
    # C[2,0]+=A[2,2]*B[2,0], C[2,1]+=A[2,2]*B[2,1]
    
    U[8][col] = 1
    V[6][col] = 1
    W[6][col] = 1
    col += 1
    
    U[8][col] = 1
    V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # a2T*b1 (rank 2)
    # a2T = U[6], U[7], b1 = V[2], V[5]
    # C[2,2]+=A[2,0]*B[0,2]+A[2,1]*B[1,2]
    # W[2*3+2]=W[8]
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[8][col] = 1
    col += 1
    
    # a33*b33 (rank 1)
    # C[2,2]+=A[2,2]*B[2,2]
    
    U[8][col] = 1
    V[8][col] = 1
    W[8][col] = 1
    col += 1
    
    assert col == 26, f"Expected 26, got {col}"
    
    # Part 2: k=3,4,5,6 using schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
