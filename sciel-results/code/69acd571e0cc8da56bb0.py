import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition with rank 44
    # 28 terms for rows 0-1 (using Strassen on 2x2 blocks)
    # 16 terms for row 2 (schoolbook)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a Strassen 2x2 product
    # X is a 2x2 submatrix of A (rows i0:i0+2, cols j0:j0+2)
    # Y is a 2x2 submatrix of B (rows j0:j0+2, cols k0:k0+2)
    # Output goes to C[i0:i0+2, k0:k0+2]
    # Strassen products:
    # M1 = (x00+x11)(y00+y11) -> c00 += M1, c11 += M1
    # M2 = (x10+x11)(y00) -> c10 += M2
    # M3 = (x00)(y01-y11) -> c01 += M3
    # M4 = (x11)(y10-y00) -> c01 += M4, c00 -= M4
    # M5 = (x00+x01)(y11) -> c10 += M5, c00 -= M5
    # M6 = (x10-x00)(y00+y01) -> c11 += M6
    # M7 = (x01-x11)(y10+y11) -> c00 -= M7
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1 = (x00+x11)(y00+y11)
        U[i0*4 + j0][col] = 1
        U[(i0+1)*4 + (j0+1)][col] = 1
        V[j0*4 + k0][col] = 1
        V[(j0+1)*4 + (k0+1)][col] = 1
        W[k0*3 + i0][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M2 = (x10+x11)(y00)
        U[(i0+1)*4 + j0][col] = 1
        U[(i0+1)*4 + (j0+1)][col] = 1
        V[j0*4 + k0][col] = 1
        W[(k0)*3 + (i0+1)][col] = 1
        col += 1
        
        # M3 = (x00)(y01-y11)
        U[i0*4 + j0][col] = 1
        V[j0*4 + (k0+1)][col] = 1
        V[(j0+1)*4 + (k0+1)][col] = -1
        W[(k0+1)*3 + i0][col] = 1
        col += 1
        
        # M4 = (x11)(y10-y00)
        U[(i0+1)*4 + (j0+1)][col] = 1
        V[(j0+1)*4 + k0][col] = 1
        V[j0*4 + k0][col] = -1
        W[(k0+1)*3 + i0][col] = 1
        W[k0*3 + i0][col] = -1
        col += 1
        
        # M5 = (x00+x01)(y11)
        U[i0*4 + j0][col] = 1
        U[i0*4 + (j0+1)][col] = 1
        V[(j0+1)*4 + (k0+1)][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        W[k0*3 + i0][col] = -1
        col += 1
        
        # M6 = (x10-x00)(y00+y01)
        U[(i0+1)*4 + j0][col] = 1
        U[i0*4 + j0][col] = -1
        V[j0*4 + k0][col] = 1
        V[j0*4 + (k0+1)][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M7 = (x01-x11)(y10+y11)
        U[i0*4 + (j0+1)][col] = 1
        U[(i0+1)*4 + (j0+1)][col] = -1
        V[(j0+1)*4 + k0][col] = 1
        V[(j0+1)*4 + (k0+1)][col] = 1
        W[k0*3 + i0][col] = -1
        col += 1
    
    # Rows 0-1 of A, using Strassen on 2x2 blocks
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 0)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 0, 2)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 2)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Row 2 of A, schoolbook
    for j in range(4):
        for k in range(4):
            U[2*4 + j][col] = 1
            V[j*4 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
