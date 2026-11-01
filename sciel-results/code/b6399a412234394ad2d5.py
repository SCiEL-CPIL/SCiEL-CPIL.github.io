import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] += val

    # --- Part 1: A1 B1 (j=0 for A, k=0 for B row index in V? No, j=0 in V is row 0 of B) ---
    # A1 is col 0 of A. Indices i*3+0.
    # B1 is row 0 of B. Indices 0*3+k = k.
    # W index k*n+i.
    
    # r=0: A00 * B00 -> C00
    set_val(U, 0, 0, 1) # i=0, j=0
    set_val(V, 0, 0, 1) # j=0, k=0
    set_val(W, 0, 0, 1) # k=0, i=0
    
    # r=1: A00 * B01 -> C01
    set_val(U, 0, 1, 1)
    set_val(V, 1, 1, 1) # j=0, k=1
    set_val(W, 1, 1, 1) # k=1, i=0
    
    # r=2: A00 * B02 -> C02
    set_val(U, 0, 2, 1)
    set_val(V, 2, 2, 1) # j=0, k=2
    set_val(W, 2, 2, 1) # k=2, i=0
    
    # r=3: A10 * B00 -> C10
    set_val(U, 3, 3, 1) # i=1, j=0
    set_val(V, 0, 3, 1)
    set_val(W, 0, 3, 1) # k=0, i=1
    
    # r=4: A10 * B01 -> C11
    set_val(U, 3, 4, 1)
    set_val(V, 1, 4, 1)
    set_val(W, 1, 4, 1) # k=1, i=1
    
    # r=5: A10 * B02 -> C12
    set_val(U, 3, 5, 1)
    set_val(V, 2, 5, 1)
    set_val(W, 2, 5, 1) # k=2, i=1

    # --- Part 2: Strassen for A2 B2a ---
    # A2 cols 1,2. Indices i*3+1, i*3+2.
    # B2a rows 1,2, cols 0,1. Indices 1*3+k, 2*3+k.
    
    # r=6: M1 = A01 * (B10 + B21)
    set_val(U, 1, 6, 1) # A01
    set_val(V, 3, 6, 1) # B10 (j=1, k=0)
    set_val(V, 7, 6, 1) # B21 (j=2, k=1)
    set_val(W, 0, 6, 1) # C00
    set_val(W, 3, 6, 1) # C11 (k=1, i=1)
    
    # r=7: M2 = (A01 + A02) * B21
    set_val(U, 1, 7, 1) # A01
    set_val(U, 2, 7, 1) # A02
    set_val(V, 7, 7, 1) # B21
    set_val(W, 2, 7, 1) # C01 (k=1, i=0)
    set_val(W, 3, 7, -1) # C11
    
    # r=8: M3 = (A11 + A12) * B10
    set_val(U, 4, 8, 1) # A11
    set_val(U, 5, 8, 1) # A12
    set_val(V, 3, 8, 1) # B10
    set_val(W, 1, 8, 1) # C10 (k=0, i=1)
    set_val(W, 3, 8, 1) # C11
    
    # r=9: M4 = A12 * (B11 - B10)
    set_val(U, 5, 9, 1) # A12
    set_val(V, 4, 9, 1) # B11 (j=1, k=1)
    set_val(V, 3, 9, -1) # B10
    set_val(W, 0, 9, 1) # C00
    set_val(W, 3, 9, 1) # C11
    
    # r=10: M5 = (A01 + A11) * (B10 + B11)
    set_val(U, 1, 10, 1) # A01
    set_val(U, 4, 10, 1) # A11
    set_val(V, 3, 10, 1) # B10
    set_val(V, 4, 10, 1) # B11
    set_val(W, 0, 10, -1) # C00
    set_val(W, 2, 10, 1) # C01
    
    # r=11: M6 = (A02 - A11) * (B20 + B21)
    set_val(U, 2, 11, 1) # A02
    set_val(U, 4, 11, -1) # A11
    set_val(V, 6, 11, 1) # B20 (j=2, k=0)
    set_val(V, 7, 11, 1) # B21
    set_val(W, 1, 11, 1) # C10
    
    # r=12: M7 = (A02 + A12) * (B21 - B11)
    set_val(U, 2, 12, 1) # A02
    set_val(U, 5, 12, 1) # A12
    set_val(V, 7, 12, 1) # B21
    set_val(V, 4, 12, -1) # B11
    set_val(W, 0, 12, 1) # C00

    # --- Part 3: Trivial for A2 B2b (col 2) ---
    # B2b rows 1,2, col 2. Indices 1*3+2=8, 2*3+2=11.
    
    # r=13: A01 * B12 -> C02
    set_val(U, 1, 13, 1)
    set_val(V, 8, 13, 1)
    set_val(W, 4, 13, 1) # k=2, i=0
    
    # r=14: A02 * B22 -> C02
    set_val(U, 2, 14, 1)
    set_val(V, 11, 14, 1)
    set_val(W, 4, 14, 1)
    
    # r=15: A11 * B12 -> C12
    set_val(U, 4, 15, 1)
    set_val(V, 8, 15, 1)
    set_val(W, 5, 15, 1) # k=2, i=1
    
    # r=16: A12 * B22 -> C12
    set_val(U, 5, 16, 1)
    set_val(V, 11, 16, 1)
    set_val(W, 5, 16, 1)

    return {"U": U, "V": V, "W": W}
