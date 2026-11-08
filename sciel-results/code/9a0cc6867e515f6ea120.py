import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 90
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # We want to replace the 2x2x2 block at (rows 0-1, cols 0-1) with Strassen.
    # The schoolbook indices for this block are:
    # i in [0,1], j in [0,1], k in [0,1]
    # There are 8 such columns.
    # Let's identify their column indices.
    # col = i*m + j*p + k? No, the loop order was i, j, k.
    # col index = (i * m + j) * p + k ? No.
    # Loop:
    # i=0, j=0, k=0 -> col 0
    # i=0, j=0, k=1 -> col 1
    # ...
    # i=0, j=1, k=0 -> col 2
    # ...
    # i=1, j=0, k=0 -> col 6 (since m=5, p=6? No, p=6. j=0, k=0..5. j=1, k=0..5)
    # Let's trace:
    # i=0:
    #   j=0: k=0..5 (cols 0-5)
    #   j=1: k=0..5 (cols 6-11)
    # i=1:
    #   j=0: k=0..5 (cols 12-17)
    #   j=1: k=0..5 (cols 18-23)
    
    # We want to replace the terms where i in [0,1], j in [0,1], k in [0,1].
    # These are:
    # (0,0,0) -> 0
    # (0,0,1) -> 1
    # (0,1,0) -> 6
    # (0,1,1) -> 7
    # (1,0,0) -> 12
    # (1,0,1) -> 13
    # (1,1,0) -> 18
    # (1,1,1) -> 19
    
    # We will zero out these 8 columns in U, V, W.
    # And then add 7 new columns for Strassen.
    # To keep the matrix shapes consistent, we can just reuse the last 8 columns?
    # Or better, we can just modify the existing 8 columns to become 7 Strassen terms and 1 zero term?
    # If we set one column to all zeros, the rank effectively decreases by 1?
    # The evaluator checks if the decomposition is exact. A zero column is valid (contributes 0).
    # So we can map the 8 schoolbook columns to 7 Strassen terms and 1 zero term.
    
    # Strassen terms for 2x2x2:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # m1 = a11*b11
    # m2 = a21*(b11+b12)
    # m3 = (a11+a12)*b22
    # m4 = a22*(b21-b11)
    # m5 = (a11+a21)*b12
    # m6 = (a12-a21)*(b21+b22)
    # m7 = (a22-a11)*(b11+b21)
    
    # C11 = m1 + m4 - m7
    # C12 = m3 + m5
    # C21 = m2 + m6
    # C22 = m1 + m3 - m2 + m6
    
    # We need to express these in terms of the global U, V, W.
    # The 8 columns we are replacing correspond to:
    # c0: i=0,j=0,k=0 (A_00, B_00, C_00)
    # c1: i=0,j=0,k=1 (A_00, B_01, C_01)
    # c2: i=0,j=1,k=0 (A_01, B_10, C_10)
    # c3: i=0,j=1,k=1 (A_01, B_11, C_11)
    # c4: i=1,j=0,k=0 (A_10, B_00, C_20) -- Wait, i=1 is row 1 of A.
    # The indices in U are i*m+j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 5
    # i=1, j=1 -> 6
    
    # Indices in V are j*p+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 6
    # j=1, k=1 -> 7
    
    # Indices in W are k*n+i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 3
    # k=0, i=1 -> 1
    # k=1, i=1 -> 4
    
    # Let's map the 8 columns to the 7 Strassen terms.
    # We will use columns 0,1,6,7,12,13,18,19.
    # We will assign:
    # Col 0 -> m1
    # Col 1 -> m2
    # Col 6 -> m3
    # Col 7 -> m4
    # Col 12 -> m5
    # Col 13 -> m6
    # Col 18 -> m7
    # Col 19 -> Zero
    
    # Define the values for U, V, W for these columns.
    
    # m1: a11*b11 -> A_00 * B_00 -> C_00
    # U[0][0] = 1 (A_00)
    # V[0][0] = 1 (B_00)
    # W[0][0] = 1 (C_00)
    
    # m2: a21*(b11+b12) -> A_10 * (B_00 + B_01) -> C_10 (C_20 in 0-indexed? No, C_21 in 1-indexed is C_1,0 in 0-indexed?
    # A row 1 is i=1. B col 0 is k=0. So C_1,0.
    # W index for C_1,0 is k*n+i = 0*3+1 = 1.
    # U index for A_10 is i*m+j = 1*5+0 = 5.
    # V index for B_00 is j*p+k = 0*6+0 = 0.
    # V index for B_01 is j*p+k = 0*6+1 = 1.
    # So for Col 1 (m2):
    # U[5][1] = 1
    # V[0][1] = 1
    # V[1][1] = 1
    # W[1][1] = 1
    
    # m3: (a11+a12)*b22 -> (A_00 + A_01) * B_11 -> C_01
    # C_01 is i=0, k=1. W index 1*3+0 = 3.
    # A_00 is U[0]. A_01 is U[1].
    # B_11 is V[7] (j=1, k=1).
    # So for Col 6 (m3):
    # U[0][6] = 1
    # U[1][6] = 1
    # V[7][6] = 1
    # W[3][6] = 1
    
    # m4: a22*(b21-b11) -> A_11 * (B_10 - B_00) -> C_11
    # C_11 is i=1, k=1. W index 1*3+1 = 4.
    # A_11 is U[6] (i=1, j=1).
    # B_10 is V[6] (j=1, k=0).
    # B_00 is V[0] (j=0, k=0).
    # So for Col 7 (m4):
    # U[6][7] = 1
    # V[6][7] = 1
    # V[0][7] = -1
    # W[4][7] = 1
    
    # m5: (a11+a21)*b12 -> (A_00 + A_10) * B_01 -> C_01? No.
    # Strassen: C12 = m3 + m5.
    # m5 = (a11+a21)*b12.
    # A_00 + A_10. B_01.
    # Result is C_01?
    # Wait, C12 is row 1, col 2. i=0, k=1.
    # Yes, W index 3.
    # So for Col 12 (m5):
    # U[0][12] = 1
    # U[5][12] = 1
    # V[1][12] = 1
    # W[3][12] = 1
    
    # m6: (a12-a21)*(b21+b22) -> (A_01 - A_10) * (B_10 + B_11) -> C_11?
    # Strassen: C22 = m1 + m3 - m2 + m6.
    # C22 is row 2, col 2. i=1, k=1. W index 4.
    # So for Col 13 (m6):
    # U[1][13] = 1
    # U[5][13] = -1
    # V[6][13] = 1
    # V[7][13] = 1
    # W[4][13] = 1
    
    # m7: (a22-a11)*(b11+b21) -> (A_11 - A_00) * (B_00 + B_10) -> C_11?
    # Strassen: C11 = m1 + m4 - m7.
    # C11 is row 1, col 1. i=0, k=0. W index 0.
    # So for Col 18 (m7):
    # U[6][18] = 1
    # U[0][18] = -1
    # V[0][18] = 1
    # V[6][18] = 1
    # W[0][18] = 1
    
    # Col 19: Zero
    # U, V, W all 0 for col 19.
    
    # Now we must clear the original values in these columns.
    # The original values were 1s.
    # We need to set them to 0 first, then assign the new values.
    
    # Columns to clear: 0, 1, 6, 7, 12, 13, 18, 19
    cols_to_clear = [0, 1, 6, 7, 12, 13, 18, 19]
    for c in cols_to_clear:
        for i in range(n*m):
            U[i][c] = 0
        for j in range(m*p):
            V[j][c] = 0
        for k in range(p*n):
            W[k*n+c//1][c] = 0 # Wait, W index is row.
            # W is (p*n, rank).
            # W[row][col]
            # So W[r][c] = 0 for all r.
            for r in range(p*n):
                W[r][c] = 0

    # Assign new values
    # m1 (Col 0)
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # m2 (Col 1)
    U[5][1] = 1
    V[0][1] = 1
    V[1][1] = 1
    W[1][1] = 1
    
    # m3 (Col 6)
    U[0][6] = 1
    U[1][6] = 1
    V[7][6] = 1
    W[3][6] = 1
    
    # m4 (Col 7)
    U[6][7] = 1
    V[6][7] = 1
    V[0][7] = -1
    W[4][7] = 1
    
    # m5 (Col 12)
    U[0][12] = 1
    U[5][12] = 1
    V[1][12] = 1
    W[3][12] = 1
    
    # m6 (Col 13)
    U[1][13] = 1
    U[5][13] = -1
    V[6][13] = 1
    V[7][13] = 1
    W[4][13] = 1
    
    # m7 (Col 18)
    U[6][18] = 1
    U[0][18] = -1
    V[0][18] = 1
    V[6][18] = 1
    W[0][18] = 1
    
    # Col 19 is already 0.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
