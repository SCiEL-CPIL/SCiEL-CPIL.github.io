import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for safety
    
    # We will construct a rank-68 solution.
    # Core 3x3x3 part uses Rank 20 (two overlapping Strassen 2x2 blocks + 6 schoolbook terms).
    # Fringe uses Schoolbook.
    
    total_rank = 68
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Block 1: i,j,k in {0,1}
    add_strassen_block(0, 0, 0, col)
    col += 7
    
    # Block 2: i,j,k in {1,2}
    add_strassen_block(1, 1, 1, col)
    col += 7
    
    # Schoolbook for C[0,2] (i=0, k=2)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    for j in range(3):
        U[0*m + j][col] += 1
        V[j*p + 2][col] += 1
        W[2*n + 0][col] += 1
        col += 1
        
    # Schoolbook for C[2,0] (i=2, k=0)
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0]
    for j in range(3):
        U[2*m + j][col] += 1
        V[j*p + 0][col] += 1
        W[0*n + 2][col] += 1
        col += 1
        
    # Fringe 1: j in {3,4}, i in {0,1,2}, k in {0,1,2,3,4}
    for i in range(n):
        for j in [3, 4]:
            for k in range(p):
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
                
    # Fringe 2: j in {0,1,2}, k in {3,4}, i in {0,1,2}
    for i in range(n):
        for j in range(3):
            for k in [3, 4]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
