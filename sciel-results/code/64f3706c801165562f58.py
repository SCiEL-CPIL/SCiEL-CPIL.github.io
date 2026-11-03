import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 3 (index 2) of A (1x4) times B (4x4)
    # Schoolbook, rank 16
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # Part 2: Rows 1-2 (indices 0,1) of A (2x4) times B (4x4)
    # Split into 4 2x2 multiplications using Strassen
    # A = [A11, A12]
    # B = [[B11, B12], [B21, B22]]
    # C1 = A11*B11 + A12*B21
    # C2 = A11*B12 + A12*B22
    
    # Helper to add Strassen block
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map local indices to global
        # Local A indices: (0,0), (0,1), (1,0), (1,1)
        # Global A indices: (a_r + lr, a_c + lc)
        
        # M1
        U[(a_r+0)*m + (a_c+0)][start_col] += 1
        U[(a_r+1)*m + (a_c+1)][start_col] += 1
        V[(b_r+0)*p + (b_c+0)][start_col] += 1
        V[(b_r+1)*p + (b_c+1)][start_col] += 1
        W[(b_c+0)*n + (c_r+0)][start_col] += 1
        W[(b_c+0)*n + (c_r+1)][start_col] += 1
        W[(b_c+1)*n + (c_r+0)][start_col] += 1
        W[(b_c+1)*n + (c_r+1)][start_col] += 1
        start_col += 1
        
        # M2
        U[(a_r+1)*m + (a_c+0)][start_col] += 1
        U[(a_r+1)*m + (a_c+1)][start_col] += 1
        V[(b_r+0)*p + (b_c+0)][start_col] += 1
        W[(b_c+0)*n + (c_r+1)][start_col] += 1
        W[(b_c+1)*n + (c_r+1)][start_col] += 1
        start_col += 1
        
        # M3
        U[(a_r+0)*m + (a_c+0)][start_col] += 1
        V[(b_r+0)*p + (b_c+1)][start_col] += 1
        V[(b_r+1)*p + (b_c+1)][start_col] -= 1
        W[(b_c+1)*n + (c_r+0)][start_col] += 1
        W[(b_c+1)*n + (c_r+1)][start_col] += 1
        start_col += 1
        
        # M4
        U[(a_r+1)*m + (a_c+1)][start_col] += 1
        V[(b_r+1)*p + (b_c+0)][start_col] += 1
        V[(b_r+0)*p + (b_c+0)][start_col] -= 1
        W[(b_c+0)*n + (c_r+0)][start_col] += 1
        W[(b_c+1)*n + (c_r+0)][start_col] += 1
        start_col += 1
        
        # M5
        U[(a_r+0)*m + (a_c+0)][start_col] += 1
        U[(a_r+0)*m + (a_c+1)][start_col] += 1
        V[(b_r+1)*p + (b_c+1)][start_col] += 1
        W[(b_c+1)*n + (c_r+0)][start_col] -= 1
        W[(b_c+1)*n + (c_r+1)][start_col] -= 1
        start_col += 1
        
        # M6
        U[(a_r+1)*m + (a_c+0)][start_col] += 1
        U[(a_r+0)*m + (a_c+0)][start_col] -= 1
        V[(b_r+0)*p + (b_c+0)][start_col] += 1
        V[(b_r+0)*p + (b_c+1)][start_col] += 1
        W[(b_c+1)*n + (c_r+1)][start_col] += 1
        start_col += 1
        
        # M7
        U[(a_r+0)*m + (a_c+1)][start_col] += 1
        U[(a_r+1)*m + (a_c+1)][start_col] -= 1
        V[(b_r+1)*p + (b_c+0)][start_col] += 1
        V[(b_r+1)*p + (b_c+1)][start_col] += 1
        W[(b_c+0)*n + (c_r+0)][start_col] += 1
        W[(b_c+0)*n + (c_r+1)][start_col] += 1
        start_col += 1
        
        return start_col

    # Block 1: A11 * B11 -> C1 (cols 0,1)
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A12 * B21 -> C1 (cols 0,1)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    
    # Block 3: A11 * B12 -> C2 (cols 2,3)
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    
    # Block 4: A12 * B22 -> C2 (cols 2,3)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    
    return {"U": U, "V": V, "W": W}
