import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # i_off, j_off, k_off are the global starting indices for i, j, k
    # col_start is the starting column index
    def add_strassen(i_off, j_off, k_off, col_start):
        # Local indices 0,1 map to i_off+0, i_off+1 etc
        # U indices: (i_off + di) * m + (j_off + dj)
        # V indices: (j_off + dj) * p + (k_off + dk)
        # W indices: (k_off + dk) * n + (i_off + di)
        
        # M1
        c = col_start
        U[i_off*4 + j_off][c] += 1
        U[(i_off+1)*4 + (j_off+1)][c] += 1
        V[(j_off)*5 + k_off][c] += 1
        V[(j_off+1)*5 + (k_off+1)][c] += 1
        W[k_off*2 + i_off][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] += 1
        
        # M2
        c = col_start + 1
        U[(i_off+1)*4 + j_off][c] += 1
        U[(i_off+1)*4 + (j_off+1)][c] += 1
        V[j_off*5 + k_off][c] += 1
        W[(k_off+1)*2 + (i_off+1)][c] += 1 # C10? No, C11 is (k=1, i=1) -> 1*2+1=3? 
        # Wait, W index is k*n+i.
        # C10 is k=1, i=0 -> 1*2+0 = 2.
        # C11 is k=1, i=1 -> 1*2+1 = 3.
        # M2 contributes to C10 (+1) and C11 (-1).
        W[k_off*2 + (i_off+1)][c] += 1 # C10: k=1, i=1? No.
        # Let's re-verify W indices for Strassen.
        # C00: k=0, i=0 -> 0
        # C01: k=0, i=1 -> 1
        # C10: k=1, i=0 -> 2
        # C11: k=1, i=1 -> 3
        
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(+1)
        # M7: C00(+1)
        
        # I will rewrite the helper carefully.
        
    # I'll just write the full logic in the main function to avoid confusion.
    
    # Sub 1a: i 0-1, j 0-1, k 0-1. Cols 0-6
    add_strassen_block(0, 0, 0, 0, U, V, W, n, m, p)
    # Sub 1b: i 0-1, j 2-3, k 0-1. Cols 7-13
    add_strassen_block(0, 2, 0, 7, U, V, W, n, m, p)
    # Sub 2a: i 0-1, j 0-1, k 2-3. Cols 14-20
    add_strassen_block(0, 0, 2, 14, U, V, W, n, m, p)
    # Sub 2b: i 0-1, j 2-3, k 2-3. Cols 21-27
    add_strassen_block(0, 2, 2, 21, U, V, W, n, m, p)
    
    # Sub 3a: i 0-1, j 0-1, k 4. Cols 28-31
    add_standard_block(0, 0, 4, 28, U, V, W, n, m, p)
    # Sub 3b: i 0-1, j 2-3, k 4. Cols 32-35
    add_standard_block(0, 2, 4, 32, U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

def add_strassen_block(i_off, j_off, k_off, col_start, U, V, W, n, m, p):
    # Local 2x2x2
    # M1
    c = col_start
    U[i_off*m + j_off][c] += 1
    U[(i_off+1)*m + (j_off+1)][c] += 1
    V[j_off*p + k_off][c] += 1
    V[(j_off+1)*p + (k_off+1)][c] += 1
    W[k_off*n + i_off][c] += 1
    W[(k_off+1)*n + (i_off+1)][c] += 1
    
    # M2
    c = col_start + 1
    U[(i_off+1)*m + j_off][c] += 1
    U[(i_off+1)*m + (j_off+1)][c] += 1
    V[j_off*p + k_off][c] += 1
    W[(k_off+1)*n + i_off][c] += 1 # C10
    W[(k_off+1)*n + (i_off+1)][c] -= 1 # C11
    
    # M3
    c = col_start + 2
    U[i_off*m + j_off][c] += 1
    V[(j_off+1)*p + k_off][c] += 1
    V[(j_off+1)*p + (k_off+1)][c] -= 1
    W[k_off*n + (i_off+1)][c] += 1 # C01
    W[(k_off+1)*n + (i_off+1)][c] += 1 # C11
    
    # M4
    c = col_start + 3
    U[(i_off+1)*m + (j_off+1)][c] += 1
    V[j_off*p + (k_off+1)][c] += 1
    V[j_off*p + k_off][c] -= 1
    W[k_off*n + i_off][c] += 1 # C00
    W[(k_off+1)*n + i_off][c] += 1 # C10
    
    # M5
    c = col_start + 4
    U[i_off*m + j_off][c] += 1
    U[(i_off+1)*m + j_off][c] += 1
    V[(j_off+1)*p + (k_off+1)][c] += 1
    W[k_off*n + i_off][c] -= 1 # C00
    W[k_off*n + (i_off+1)][c] += 1 # C01
    
    # M6
    c = col_start + 5
    U[(i_off+1)*m + j_off][c] += 1
    U[i_off*m + j_off][c] -= 1
    V[j_off*p + k_off][c] += 1
    V[j_off*p + (k_off+1)][c] += 1
    W[(k_off+1)*n + (i_off+1)][c] += 1 # C11
    
    # M7
    c = col_start + 6
    U[i_off*m + (j_off+1)][c] += 1
    U[(i_off+1)*m + (j_off+1)][c] -= 1
    V[(j_off+1)*p + k_off][c] += 1
    V[(j_off+1)*p + (k_off+1)][c] += 1
    W[k_off*n + i_off][c] += 1 # C00

def add_standard_block(i_off, j_off, k_off, col_start, U, V, W, n, m, p):
    # Local 2x2x1
    # M1: A00 B00 -> C00
    c = col_start
    U[i_off*m + j_off][c] += 1
    V[j_off*p + k_off][c] += 1
    W[k_off*n + i_off][c] += 1
    
    # M2: A01 B10 -> C00
    c = col_start + 1
    U[i_off*m + (j_off+1)][c] += 1
    V[(j_off+1)*p + k_off][c] += 1
    W[k_off*n + i_off][c] += 1
    
    # M3: A10 B00 -> C10
    c = col_start + 2
    U[(i_off+1)*m + j_off][c] += 1
    V[j_off*p + k_off][c] += 1
    W[k_off*n + (i_off+1)][c] += 1
    
    # M4: A11 B10 -> C10
    c = col_start + 3
    U[(i_off+1)*m + (j_off+1)][c] += 1
    V[(j_off+1)*p + k_off][c] += 1
    W[k_off*n + (i_off+1)][c] += 1

