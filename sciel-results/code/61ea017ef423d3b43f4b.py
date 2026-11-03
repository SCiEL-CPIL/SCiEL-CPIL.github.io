import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_laderman_block(b_col_offset):
        nonlocal col
        # b_col_offset is 0 or 3
        # c_col_offset is the same
        
        # 1
        U[0*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+0+b_col_offset][col] += 1; V[1*p+1+b_col_offset][col] += 1
        W[0*n+0][col] += 1; W[1*n+0][col] += 1; W[0*n+1][col] += 1; W[1*n+1][col] += 1
        col += 1
        
        # 2
        U[1*m+0][col] += 1; U[1*m+1][col] += 1
        V[0*p+0+b_col_offset][col] += 1
        W[0*n+1][col] += 1
        col += 1
        
        # 3
        U[0*m+0][col] += 1
        V[0*p+1+b_col_offset][col] += 1; V[1*p+1+b_col_offset][col] -= 1
        W[1*n+0][col] += 1; W[1*n+1][col] += 1
        col += 1
        
        # 4
        U[1*m+1][col] += 1
        V[1*p+0+b_col_offset][col] += 1; V[0*p+0+b_col_offset][col] -= 1
        W[0*n+1][col] += 1; W[0*n+0][col] += 1
        col += 1
        
        # 5
        U[0*m+0][col] += 1; U[0*m+1][col] += 1
        V[1*p+1+b_col_offset][col] += 1
        W[0*n+1][col] += 1; W[1*n+1][col] += 1
        col += 1
        
        # 6
        U[1*m+0][col] += 1; U[0*m+0][col] -= 1
        V[0*p+0+b_col_offset][col] += 1; V[0*p+1+b_col_offset][col] += 1
        W[0*n+1][col] += 1; W[1*n+1][col] += 1
        col += 1
        
        # 7
        U[0*m+1][col] += 1; U[1*m+1][col] -= 1
        V[1*p+0+b_col_offset][col] += 1; V[1*p+1+b_col_offset][col] += 1
        W[0*n+0][col] += 1; W[1*n+1][col] += 1
        col += 1
        
        # 8
        U[2*m+0][col] += 1; U[2*m+1][col] += 1; U[2*m+2][col] += 1
        V[0*p+0+b_col_offset][col] += 1; V[0*p+1+b_col_offset][col] += 1; V[0*p+2+b_col_offset][col] += 1
        W[0*n+2][col] += 1; W[1*n+2][col] += 1; W[2*n+2][col] += 1
        col += 1
        
        # 9
        U[2*m+0][col] += 1; U[2*m+1][col] += 1
        V[0*p+0+b_col_offset][col] += 1; V[0*p+1+b_col_offset][col] += 1
        W[0*n+2][col] += 1; W[1*n+2][col] += 1
        col += 1
        
        # 10
        U[2*m+0][col] += 1; U[2*m+2][col] += 1
        V[0*p+0+b_col_offset][col] += 1; V[0*p+2+b_col_offset][col] += 1
        W[0*n+2][col] += 1; W[2*n+2][col] += 1
        col += 1
        
        # 11
        U[2*m+1][col] += 1; U[2*m+2][col] += 1
        V[0*p+1+b_col_offset][col] += 1; V[0*p+2+b_col_offset][col] += 1
        W[1*n+2][col] += 1; W[2*n+2][col] += 1
        col += 1
        
        # 12
        U[2*m+0][col] += 1
        V[0*p+0+b_col_offset][col] += 1; V[0*p+2+b_col_offset][col] += 1; V[0*p+1+b_col_offset][col] -= 1
        W[0*n+2][col] += 1
        col += 1
        
        # 13
        U[2*m+1][col] += 1
        V[0*p+1+b_col_offset][col] += 1; V[0*p+0+b_col_offset][col] += 1; V[0*p+2+b_col_offset][col] -= 1
        W[1*n+2][col] += 1
        col += 1
        
        # 14
        U[2*m+2][col] += 1
        V[0*p+2+b_col_offset][col] += 1; V[0*p+1+b_col_offset][col] += 1; V[0*p+0+b_col_offset][col] -= 1
        W[2*n+2][col] += 1
        col += 1
        
        # 15
        U[0*m+0][col] += 1; U[0*m+1][col] += 1; U[0*m+2][col] += 1
        V[2*p+0+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] += 1; V[2*p+2+b_col_offset][col] += 1
        W[0*n+0][col] += 1; W[1*n+0][col] += 1; W[2*n+0][col] += 1
        col += 1
        
        # 16
        U[0*m+0][col] += 1; U[0*m+1][col] += 1
        V[2*p+0+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] += 1
        W[0*n+0][col] += 1; W[1*n+0][col] += 1
        col += 1
        
        # 17
        U[0*m+0][col] += 1; U[0*m+2][col] += 1
        V[2*p+0+b_col_offset][col] += 1; V[2*p+2+b_col_offset][col] += 1
        W[0*n+0][col] += 1; W[2*n+0][col] += 1
        col += 1
        
        # 18
        U[0*m+1][col] += 1; U[0*m+2][col] += 1
        V[2*p+1+b_col_offset][col] += 1; V[2*p+2+b_col_offset][col] += 1
        W[1*n+0][col] += 1; W[2*n+0][col] += 1
        col += 1
        
        # 19
        U[0*m+0][col] += 1
        V[2*p+0+b_col_offset][col] += 1; V[2*p+2+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] -= 1
        W[0*n+0][col] += 1
        col += 1
        
        # 20
        U[0*m+1][col] += 1
        V[2*p+1+b_col_offset][col] += 1; V[2*p+0+b_col_offset][col] += 1; V[2*p+2+b_col_offset][col] -= 1
        W[1*n+0][col] += 1
        col += 1
        
        # 21
        U[0*m+2][col] += 1
        V[2*p+2+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] += 1; V[2*p+0+b_col_offset][col] -= 1
        W[2*n+0][col] += 1
        col += 1
        
        # 22
        U[1*m+0][col] += 1
        V[1*p+0+b_col_offset][col] += 1; V[2*p+0+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] -= 1
        W[0*n+1][col] += 1
        col += 1
        
        # 23
        U[1*m+1][col] += 1
        V[1*p+1+b_col_offset][col] += 1; V[2*p+1+b_col_offset][col] += 1; V[2*p+0+b_col_offset][col] -= 1
        W[1*n+1][col] += 1
        col += 1

    # Block 1: B cols 0-2
    add_laderman_block(0)
    
    # Block 2: B cols 3-5
    add_laderman_block(3)
    
    # Block 3: B col 6
    # 9 terms
    for i in range(3):
        for j in range(3):
            U[i*m+j][col] = 1
            V[j*p+6][col] = 1
            W[6*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
