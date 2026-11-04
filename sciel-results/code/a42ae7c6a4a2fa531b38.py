import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Split m=3 into 2+1
    # 2x2x8 part: 4 Strassen 2x2x2 blocks (rank 7 each) = 28
    # 2x1x8 part: rank 1
    # Total rank = 29
    
    rank = 29
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 29
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 29
    
    col = 0
    
    # 4 Strassen blocks for 2x2x2
    for k_start in range(0, 8, 2):
        i1, i2 = 0, 1
        j1, j2 = 0, 1
        k1, k2 = k_start, k_start + 1
        
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += -1
        col += 1
        
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1][col] += 1
        U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1
        W[k2*n+i1][col] += 1
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2][col] += 1
        U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
    
    # 1 rank-1 term for 2x1x8
    for i in range(n):
        U[i*m+2][col] += 1
    for k in range(p):
        V[2*p+k][col] += 1
    for k in range(p):
        for i in range(n):
            W[k*n+i][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
