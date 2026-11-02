import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will use the block decomposition
    # A is 3x3, B is 3x8
    # Split A into A22 (2x2), a21 (2x1), a12 (1x2), a11 (1x1)
    # Split B into B28 (2x8), b18 (1x8)
    
    # Rank components:
    # 1. A22 * B28: Rank 28 (using Strassen recursively)
    # 2. a21 * b18: Rank 1
    # 3. a12 * B28: Rank 16 (schoolbook 1x2 by 2x8)
    # 4. a11 * b18: Rank 1
    
    total_rank = 28 + 1 + 16 + 1
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen term for 2x2x2
    # offsets: i0, j0, k0 for A, B, C
    def add_strassen_2x2x2(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*3 + 0][col] += 1
        U[i0*3 + 2][col] += 1 # Wait, A is 3x3. A22 is at rows 0,1 cols 0,1.
        # A indices: i in {0,1}, j in {0,1}
        # So A[i0+0, j0+0] and A[i0+1, j0+1]
        U[i0*3 + 0][col] = 1
        U[(i0+1)*3 + 1][col] = 1
        
        V[j0*8 + 0][col] = 1
        V[(j0+1)*8 + 1][col] = 1
        
        W[k0*3 + i0][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[(i0+1)*3 + 0][col] = 1
        U[(i0+1)*3 + 1][col] = 1
        V[j0*8 + 0][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        W[k0*3 + (i0+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[i0*3 + 0][col] = 1
        V[j0*8 + 1][col] = 1
        V[(j0+1)*8 + 1][col] = -1
        W[k0*3 + i0][col] = 1
        W[(k0+1)*3 + i0][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[(i0+1)*3 + 1][col] = 1
        V[(j0+1)*8 + 0][col] = 1
        V[j0*8 + 0][col] = -1
        W[k0*3 + (i0+1)][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[i0*3 + 0][col] = 1
        U[i0*3 + 1][col] = 1
        V[(j0+1)*8 + 1][col] = 1
        W[k0*3 + i0][col] = -1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i0+1)*3 + 0][col] = 1
        U[i0*3 + 0][col] = -1
        V[j0*8 + 0][col] = 1
        V[j0*8 + 1][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i0*3 + 1][col] = 1
        U[(i0+1)*3 + 1][col] = -1
        V[(j0+1)*8 + 0][col] = 1
        V[(j0+1)*8 + 1][col] = 1
        W[k0*3 + i0][col] = 1
        W[(k0+1)*3 + (i0+1)][col] = 1
        col += 1

    # 1. A22 * B28
    # B28 is 2x8. Split into two 2x4.
    # Split each 2x4 into two 2x2.
    # So 4 blocks of 2x2.
    for k_block in range(4):
        add_strassen_2x2x2(0, 0, k_block * 2)
        
    # 2. a21 * b18
    # a21 is A[0:2, 2] (col 2 of A, rows 0,1)
    # b18 is B[2, 0:8] (row 2 of B, cols 0..7)
    # Rank 1
    for i in range(2):
        for k in range(8):
            # Term A[i, 2] B[2, k] contributes to C[i, k]
            # U[i*3 + 2] = 1
            # V[2*8 + k] = 1
            # W[k*3 + i] = 1
            # But this is rank 8 if done schoolbook.
            # Can we do rank 1?
            # We need sum_r U[i,2,r] V[2,k,r] W[k,i,r] = 1 for all i,k?
            # No, it's A[i,2] B[2,k].
            # If we set U[i,2,r]=1, V[2,k,r]=1, W[k,i,r]=1 for a single r,
            # then sum is 1.
            # But this adds A[i,2]B[2,k] to C[i,k].
            # This is exactly the term we want.
            # So we can use 1 column?
            # Wait, U[i,2,r] must be 1 for all i?
            # If U[0,2,r]=1 and U[1,2,r]=1, then it works for i=0 and i=1.
            # If V[2,0,r]=1 ... V[2,7,r]=1, it works for all k.
            # If W[k,0,r]=1 and W[k,1,r]=1, it works for all i,k.
            # So yes, rank 1.
            pass
    
    # Add the single column for a21*b18
    for i in range(2):
        U[i*3 + 2][col] = 1
    for k in range(8):
        V[2*8 + k][col] = 1
    for i in range(2):
        for k in range(8):
            W[k*3 + i][col] = 1
    col += 1
    
    # 3. a12 * B28
    # a12 is A[2, 0:2] (row 2 of A, cols 0,1)
    # B28 is B[0:2, 0:8]
    # Schoolbook 1x2 by 2x8. Rank 16.
    for j in range(2):
        for k in range(8):
            U[2*3 + j][col] = 1
            V[j*8 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1
            
    # 4. a11 * b18
    # A[2,2] B[2,k]
    # Rank 1
    U[2*3 + 2][col] = 1
    for k in range(8):
        V[2*8 + k][col] = 1
    for k in range(8):
        W[k*3 + 2][col] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}

