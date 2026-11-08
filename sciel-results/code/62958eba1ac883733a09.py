import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition using Strassen on 2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Define the 7 multiplications for one Strassen block
    # Each entry is (coeff_A, coeff_B, coeff_C)
    # We need to map local indices to global indices
    
    # Blocks:
    # A1: cols 0,1. A2: cols 2,3.
    # B1: rows 0,1. B2: rows 2,3.
    # B11: cols 0,1. B12: cols 2,3. B13: cols 4,5.
    # B21: cols 0,1. B22: cols 2,3. B23: cols 4,5.
    
    blocks = [
        (0, 0), # A1, B11
        (0, 2), # A1, B12
        (0, 4), # A1, B13
        (2, 0), # A2, B21
        (2, 2), # A2, B22
        (2, 4), # A2, B23
    ]
    
    col = 0
    for a_col_start, b_col_start in blocks:
        # A local indices: 0,1 -> global a_col_start, a_col_start+1
        # B local indices: 0,1 -> global rows (if A1 then 0,1; if A2 then 2,3)
        # Wait, A1 is cols 0,1 of A. B1 is rows 0,1 of B.
        # If block is A1, B rows are 0,1.
        # If block is A2, B rows are 2,3.
        
        if a_col_start == 0:
            b_row_start = 0
        else:
            b_row_start = 2
            
        # Local A indices: 0,1
        # Local B indices: 0,1
        # Global A indices: (i, a_col_start + j)
        # Global B indices: (b_row_start + j, b_col_start + k)
        
        # Strassen terms
        # m1: (A00+A11)(B00+B11)
        U[a_col_start][col] = 1
        U[a_col_start + 1][col] = 1
        V[b_row_start * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[b_col_start * n + 0][col] = 1
        W[(b_col_start + 1) * n + 1][col] = 1
        col += 1
        
        # m2: (A10+A11)B00
        U[a_col_start + 1][col] = 1
        V[b_row_start * p + b_col_start][col] = 1
        W[b_col_start * n + 1][col] = 1
        col += 1
        
        # m3: A00(B01-B11)
        U[a_col_start][col] = 1
        V[b_row_start * p + (b_col_start + 1)][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = -1
        W[(b_col_start + 1) * n + 0][col] = 1
        col += 1
        
        # m4: A11(B10-B00)
        U[a_col_start + 1][col] = 1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        V[b_row_start * p + b_col_start][col] = -1
        W[b_col_start * n + 1][col] = 1
        col += 1
        
        # m5: (A00+A01)B11
        U[a_col_start][col] = 1
        U[a_col_start + 1][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[(b_col_start + 1) * n + 0][col] = 1
        col += 1
        
        # m6: (A10-A00)(B00+B01)
        U[a_col_start + 1][col] = 1
        U[a_col_start][col] = -1
        V[b_row_start * p + b_col_start][col] = 1
        V[b_row_start * p + (b_col_start + 1)][col] = 1
        W[(b_col_start + 1) * n + 1][col] = 1
        col += 1
        
        # m7: (A01-A11)(B10+B11)
        U[a_col_start + 1][col] = 1
        U[a_col_start][col] = -1
        V[(b_row_start + 1) * p + b_col_start][col] = 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col] = 1
        W[b_col_start * n + 0][col] = 1
        W[(b_col_start + 1) * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
