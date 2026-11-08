import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U: u=3*i+j, V: v=5*j+k, W: w=2*k+i
    
    col = 0
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    
    col = 1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    
    col = 2
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    
    col = 3
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[0][col] = 1
    
    col = 4
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[2][col] = 1; W[0][col] = -1
    
    col = 5
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    
    col = 6
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    
    # Part 2a: Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[2:4,0:2]
    # U: u=3*i+j (i,j in {0,1}) -> {0,1,3,4}
    # V: v=5*j+k (j in {0,1}, k in {2,3}) -> {2,3,7,8}
    # W: w=2*k+i (k in {2,3}, i in {0,1}) -> {4,5,6,7}
    
    # M1: (A00+A11)(B00+B11) where B00=v2, B11=v8
    col = 7
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[5][col] = -1; W[6][col] = -1; W[7][col] = 1
    
    # M2: (A10+A11)B00
    col = 8
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = 1; W[7][col] = -1
    
    # M3: A00(B01-B11) where B01=v3, B11=v8
    col = 9
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[5][col] = 1; W[7][col] = 1
    
    # M4: A11(B10-B00) where B10=v7, B00=v2
    col = 10
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[4][col] = 1
    
    # M5: (A00+A01)B11
    col = 11
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[5][col] = 1; W[4][col] = -1
    
    # M6: (A10-A00)(B00+B01)
    col = 12
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[6][col] = 1
    
    # M7: (A01-A11)(B10+B11)
    col = 13
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[7][col] = 1
    
    # Part 2b: Schoolbook for A[0:2,0:2] * B[0:2,4:5] -> C[4:5,0:2]
    # U: u=3*i+j, V: v=5*j+4, W: w=2*4+i=8+i
    col = 14
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    
    col = 15
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    
    col = 16
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    
    col = 17
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    
    # Part 3: Schoolbook for A[0:2,2:3] * B[2:3,0:5] -> C[0:5,0:2]
    # U: u=3*i+2, V: v=10+k, W: w=2*k+i
    for i in range(2):
        for k in range(5):
            u_idx = 3 * i + 2
            v_idx = 10 + k
            w_idx = 2 * k + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
