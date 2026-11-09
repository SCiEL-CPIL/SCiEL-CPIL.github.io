import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A is 2x3: indices 0,1,2 for row 0; 3,4,5 for row 1
    # B is 3x9: indices 0..8 for row 0; 9..17 for row 1; 18..26 for row 2
    # C is 2x9: indices 0..8 for row 0; 9..17 for row 1
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:9]
    # This involves A indices 0,1,3,4
    # B indices 0..8 (row 0) and 9..17 (row 1)
    
    # We split B[0:2, 0:9] into 3 blocks of 2x3
    for block in range(3):
        # Block cols in B: 3*block to 3*block+2
        # Corresponding C cols: 3*block to 3*block+2
        
        # Strassen for first 2 cols of this block
        # B cols: 3*block, 3*block+1
        # A rows: 0, 1 (indices 0,1 for row 0; 3,4 for row 1)
        
        # M1 = (a+d)(x+w)
        # a=A[0,0]=0, d=A[1,1]=4
        # x=B[0, 3*block]=3*block, w=B[1, 3*block+1]=9+3*block+1
        U[0][col] += 1
        U[4][col] += 1
        V[3*block][col] += 1
        V[9+3*block+1][col] += 1
        # C[0, 3*block] += M1
        W[3*block][col] += 1
        # C[1, 3*block+1] += M1
        W[9+3*block+1][col] += 1
        col += 1
        
        # M2 = (c+d)x
        # c=A[1,0]=3, d=A[1,1]=4
        # x=B[0, 3*block]
        U[3][col] += 1
        U[4][col] += 1
        V[3*block][col] += 1
        # C[1, 3*block] += M2
        W[9+3*block][col] += 1
        col += 1
        
        # M3 = a(y-w)
        # a=A[0,0]=0
        # y=B[0, 3*block+1], w=B[1, 3*block+1]
        U[0][col] += 1
        V[3*block+1][col] += 1
        V[9+3*block+1][col] += -1
        # C[0, 3*block+1] += M3
        W[3*block+1][col] += 1
        # C[1, 3*block+1] += M3
        W[9+3*block+1][col] += 1
        col += 1
        
        # M4 = d(z-x)
        # d=A[1,1]=4
        # z=B[1, 3*block], x=B[0, 3*block]
        U[4][col] += 1
        V[9+3*block][col] += 1
        V[3*block][col] += -1
        # C[0, 3*block] += M4
        W[3*block][col] += 1
        # C[1, 3*block] += M4
        W[9+3*block][col] += 1
        col += 1
        
        # M5 = (a+b)w
        # a=A[0,0]=0, b=A[0,1]=1
        # w=B[1, 3*block+1]
        U[0][col] += 1
        U[1][col] += 1
        V[9+3*block+1][col] += 1
        # C[0, 3*block] -= M5
        W[3*block][col] += -1
        # C[0, 3*block+1] += M5
        W[3*block+1][col] += 1
        col += 1
        
        # M6 = (c-a)(x+z)
        # c=A[1,0]=3, a=A[0,0]=0
        # x=B[0, 3*block], z=B[1, 3*block]
        U[3][col] += 1
        U[0][col] += -1
        V[3*block][col] += 1
        V[9+3*block][col] += 1
        # C[1, 3*block+1] += M6
        W[9+3*block+1][col] += 1
        col += 1
        
        # M7 = (b-d)(y+w)
        # b=A[0,1]=1, d=A[1,1]=4
        # y=B[0, 3*block+1], w=B[1, 3*block+1]
        U[1][col] += 1
        U[4][col] += -1
        V[3*block+1][col] += 1
        V[9+3*block+1][col] += 1
        # C[0, 3*block] += M7
        W[3*block][col] += 1
        # C[0, 3*block+1] -= M7
        W[3*block+1][col] += -1
        col += 1
        
        # Schoolbook for 3rd col of this block
        # Col index in B: 3*block+2
        # Col index in C: 3*block+2
        
        # C[0, 3*block+2] = A[0,0]B[0, 3*block+2] + A[0,1]B[1, 3*block+2]
        # Term 1: A[0,0] * B[0, 3*block+2]
        U[0][col] += 1
        V[3*block+2][col] += 1
        W[3*block+2][col] += 1
        col += 1
        
        # Term 2: A[0,1] * B[1, 3*block+2]
        U[1][col] += 1
        V[9+3*block+2][col] += 1
        W[3*block+2][col] += 1
        col += 1
        
        # C[1, 3*block+2] = A[1,0]B[0, 3*block+2] + A[1,1]B[1, 3*block+2]
        # Term 3: A[1,0] * B[0, 3*block+2]
        U[3][col] += 1
        V[3*block+2][col] += 1
        W[9+3*block+2][col] += 1
        col += 1
        
        # Term 4: A[1,1] * B[1, 3*block+2]
        U[4][col] += 1
        V[9+3*block+2][col] += 1
        W[9+3*block+2][col] += 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:9]
    # A col 2: indices 2 (row 0), 5 (row 1)
    # B row 2: indices 18..26
    
    for j in range(9):
        # C[0, j] += A[0,2] * B[2, j]
        U[2][col] += 1
        V[18+j][col] += 1
        W[j][col] += 1
        col += 1
        
        # C[1, j] += A[1,2] * B[2, j]
        U[5][col] += 1
        V[18+j][col] += 1
        W[9+j][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
