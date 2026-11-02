import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Sub-problem 1: A[0:2, 0:2] x B[0:2, :] -> C[0:2, :]
    # T1: A[0,0] * B[0,k] -> C[0,k]
    U[0*m+0][col] = 1
    for k in range(p):
        V[0*p+k][col] = 1
        W[k*n+0][col] = 1
    col += 1
    # T2: A[0,1] * B[1,k] -> C[0,k]
    U[0*m+1][col] = 1
    for k in range(p):
        V[1*p+k][col] = 1
        W[k*n+0][col] = 1
    col += 1
    # T3: A[1,0] * B[0,k] -> C[1,k]
    U[1*m+0][col] = 1
    for k in range(p):
        V[0*p+k][col] = 1
        W[k*n+1][col] = 1
    col += 1
    # T4: A[1,1] * B[1,k] -> C[1,k]
    U[1*m+1][col] = 1
    for k in range(p):
        V[1*p+k][col] = 1
        W[k*n+1][col] = 1
    col += 1
    
    # Sub-problem 2: A[0:2, 2:4] x B[2:4, :] -> C[0:2, :]
    # T5: A[0,2] * B[2,k] -> C[0,k]
    U[0*m+2][col] = 1
    for k in range(p):
        V[2*p+k][col] = 1
        W[k*n+0][col] = 1
    col += 1
    # T6: A[0,3] * B[3,k] -> C[0,k]
    U[0*m+3][col] = 1
    for k in range(p):
        V[3*p+k][col] = 1
        W[k*n+0][col] = 1
    col += 1
    # T7: A[1,2] * B[2,k] -> C[1,k]
    U[1*m+2][col] = 1
    for k in range(p):
        V[2*p+k][col] = 1
        W[k*n+1][col] = 1
    col += 1
    # T8: A[1,3] * B[3,k] -> C[1,k]
    U[1*m+3][col] = 1
    for k in range(p):
        V[3*p+k][col] = 1
        W[k*n+1][col] = 1
    col += 1
    
    # Sub-problem 3: A[2:3, 0:2] x B[0:2, :] -> C[2:3, :]
    # T9: A[2,0] * B[0,k] -> C[2,k]
    U[2*m+0][col] = 1
    for k in range(p):
        V[0*p+k][col] = 1
        W[k*n+2][col] = 1
    col += 1
    # T10: A[2,1] * B[1,k] -> C[2,k]
    U[2*m+1][col] = 1
    for k in range(p):
        V[1*p+k][col] = 1
        W[k*n+2][col] = 1
    col += 1
    
    # Sub-problem 4: A[2:3, 2:4] x B[2:4, :] -> C[2:3, :]
    # T11: A[2,2] * B[2,k] -> C[2,k]
    U[2*m+2][col] = 1
    for k in range(p):
        V[2*p+k][col] = 1
        W[k*n+2][col] = 1
    col += 1
    # T12: A[2,3] * B[3,k] -> C[2,k]
    U[2*m+3][col] = 1
    for k in range(p):
        V[3*p+k][col] = 1
        W[k*n+2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
