import numpy as np
import time

DIMENSIONS = [2, 3, 8]
START_TIME = time.time()
TIME_LIMIT = 140  # Leave 10s margin

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return _rank_44_solution()
    
    return _rank_44_solution()

def _rank_44_solution():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> 2x8
    # This part corresponds to j=0 in the summation.
    # C[i,k] += A[i,0] * B[0,k]
    # Indices: U[i*3+0], V[0*8+k], W[k*2+i]
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x8
    # This part corresponds to j=1 and j=2.
    # We split B[1:3, :] into 4 blocks of 2 columns.
    # For each block, we use Strassen for the 2x2 * 2x2 multiplication.
    # A2 is 2x2: indices 1,2 (row 0) and 4,5 (row 1).
    # B2 block is 2x2.
    
    for block in range(4):
        # Block columns: block*2, block*2+1
        # B rows: 1, 2 (global indices 1*8+col, 2*8+col)
        
        # Local A indices:
        # a00 = A[0,1] -> idx 1
        # a01 = A[0,2] -> idx 2
        # a10 = A[1,1] -> idx 4
        # a11 = A[1,2] -> idx 5
        
        # Local B indices for this block:
        # b00 = B[1, block*2] -> idx 1*8 + block*2
        # b01 = B[1, block*2+1] -> idx 1*8 + block*2+1
        # b10 = B[2, block*2] -> idx 2*8 + block*2
        # b11 = B[2, block*2+1] -> idx 2*8 + block*2+1
        
        # Output C indices:
        # c00 = C[0, block*2] -> W idx (block*2)*2 + 0
        # c01 = C[0, block*2+1] -> W idx (block*2+1)*2 + 0
        # c10 = C[1, block*2] -> W idx (block*2)*2 + 1
        # c11 = C[1, block*2+1] -> W idx (block*2+1)*2 + 1
        
        # Strassen steps for 2x2 * 2x2:
        # m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1*8 + block*2][col] = 1
        W[(block*2)*2 + 1][col] = 1
        W[(block*2+1)*2 + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        V[2*8 + block*2+1][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[5][col] = 1
        V[2*8 + block*2][col] = 1
        V[1*8 + block*2][col] = -1
        W[(block*2)*2 + 0][col] = 1
        W[(block*2)*2 + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = -1
        W[(block*2+1)*2 + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1*8 + block*2][col] = 1
        V[1*8 + block*2+1][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2*8 + block*2][col] = 1
        V[2*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
