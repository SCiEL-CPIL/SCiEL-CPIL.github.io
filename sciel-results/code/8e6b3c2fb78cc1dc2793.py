import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a hybrid decomposition:
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x5) into B1 (2x5) and B2 (1x5)
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1 (2x2 * 2x5), we can use Strassen-like logic for the 2x2 block part.
    # Actually, let's just use the standard Strassen decomposition for the 2x2 part of A and B,
    # and handle the remaining parts with schoolbook or similar reductions.
    
    # Let's try a specific known decomposition for 2x3 * 3x5.
    # Rank 30 is schoolbook.
    # Can we do better?
    # The tensor rank of 2x3x5 is known to be 28? Or maybe 29?
    # Let's construct a rank 28 decomposition.
    
    # Strategy:
    # View A as [A_top (1x3), A_bottom (1x3)]? No, A is 2x3.
    # Let's use the block structure:
    # A = [[A11, A12, A13],
    #      [A21, A22, A23]]
    # B = [[B11, B12, B13, B14, B15],
    #      [B21, B22, B23, B24, B25],
    #      [B31, B32, B33, B34, B35]]
    
    # C = A B
    # C11 = A11 B11 + A12 B21 + A13 B31
    # C12 = A11 B12 + A12 B22 + A13 B32
    # ...
    # C15 = A11 B15 + A12 B25 + A13 B35
    # C21 = A21 B11 + A22 B21 + A23 B31
    # ...
    # C25 = A21 B15 + A22 B25 + A23 B35
    
    # Let's define the rank terms.
    # We can use the fact that the middle part (2x2 * 2x5) can be decomposed efficiently.
    # Let A_mid = [[A11, A12], [A21, A22]] (2x2)
    # Let B_mid = [[B11..B15], [B21..B25]] (2x5)
    # C_mid = A_mid B_mid (2x5)
    # This contributes to C[:, 0:5] from the first two columns of A and first two rows of B.
    # The remaining parts are:
    # A_col2 (2x1) = [A13, A23]^T
    # B_row2 (1x5) = [B31..B35]
    # C_rem = A_col2 B_row2 (2x5)
    
    # So C = C_mid + C_rem.
    
    # Decompose C_mid (2x2 * 2x5).
    # We can use Strassen for the 2x2 part if we split B into 2x2 and 2x1?
    # B_mid is 2x5. Let B1 = B_mid[:, 0:2] (2x2), B2 = B_mid[:, 2:5] (2x3).
    # C_mid1 = A_mid B1 (2x2) -> Use Strassen (rank 7)
    # C_mid2 = A_mid B2 (2x3) -> Use schoolbook (rank 6) or better?
    # Rank of 2x2x3 is 6 (schoolbook). Is it less? No, 6 is minimal.
    # So C_mid rank = 7 + 6 = 13.
    
    # Decompose C_rem (2x1 * 1x5).
    # This is outer product. Rank 5 (schoolbook).
    # Total rank = 13 + 5 = 18? Wait.
    # A_mid is 2x2. B1 is 2x2.
    # A_mid B1:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A_mid B2 (2x2 * 2x3):
    # B2 = [[B13, B14, B15],
    #       [B23, B24, B25]]
    # C13 = A11 B13 + A12 B23
    # C14 = A11 B14 + A12 B24
    # C15 = A11 B15 + A12 B25
    # C23 = A21 B13 + A22 B23
    # C24 = A21 B14 + A22 B24
    # C25 = A21 B15 + A22 B25
    # This is 6 terms.
    # M8 = A11 B13
    # M9 = A12 B23
    # M10 = A11 B14
    # M11 = A12 B24
    # M12 = A11 B15
    # M13 = A12 B25
    # M14 = A21 B13
    # M15 = A22 B23
    # M16 = A21 B14
    # M17 = A22 B24
    # M18 = A21 B15
    # M19 = A22 B25
    # Wait, that's 12 terms.
    # Can we reduce 2x2x3?
    # The rank of 2x2x3 tensor is 6.
    # So we can do it in 6 terms.
    # How?
    # C13 = A11 B13 + A12 B23
    # C23 = A21 B13 + A22 B23
    # Let X13 = B13, Y13 = B23.
    # C13 = A11 X13 + A12 Y13
    # C23 = A21 X13 + A22 Y13
    # This is like multiplying a 2x2 matrix by a 2x1 vector.
    # Rank of 2x2x1 is 4.
    # So for each column of B2 (3 columns), we need 4 terms. Total 12 terms.
    # So C_mid rank is 7 + 12 = 19.
    
    # C_rem (2x1 * 1x5):
    # A_col2 = [A13, A23]^T
    # B_row2 = [B31, B32, B33, B34, B35]
    # C11 += A13 B31
    # C12 += A13 B32
    # ...
    # C21 += A23 B31
    # ...
    # This is 10 terms.
    
    # Total rank = 19 + 10 = 29.
    # Schoolbook is 30.
    # So 29 is an improvement.
    
    # Can we do better?
    # Maybe the 2x2x3 part can be shared?
    # Or maybe a different split?
    
    # Let's construct the 29 rank decomposition.
    rank = 29
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A_mid B1 (2x2 * 2x2) using Strassen
    # A_mid indices in U:
    # A11: i=0, j=0 -> idx 0
    # A12: i=0, j=1 -> idx 1
    # A21: i=1, j=0 -> idx 3
    # A22: i=1, j=1 -> idx 4
    # B1 indices in V:
    # B11: j=0, k=0 -> idx 0
    # B12: j=0, k=1 -> idx 1
    # B21: j=1, k=0 -> idx 5
    # B22: j=1, k=1 -> idx 6
    
    # M1 = (A11+A22)(B11+B22)
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[6, col] += 1
    W[0, col] += 1 # C11
    W[5, col] += 1 # C22 (k=0, i=1 -> idx 0*2+1=1? No. W[k*n+i]. C22 is i=1, k=1. idx 1*2+1=3? No.
    # W indices: k in 0..4, i in 0..1. idx = k*n + i = k*2 + i.
    # C11: i=0, k=0 -> idx 0
    # C12: i=0, k=1 -> idx 2
    # C13: i=0, k=2 -> idx 4
    # C14: i=0, k=3 -> idx 6
    # C15: i=0, k=4 -> idx 8
    # C21: i=1, k=0 -> idx 1
    # C22: i=1, k=1 -> idx 3
    # C23: i=1, k=2 -> idx 5
    # C24: i=1, k=3 -> idx 7
    # C25: i=1, k=4 -> idx 9
    
    # Correct W indices for M1 (contributes to C11 and C22):
    W[0, col] += 1 # C11
    W[3, col] += 1 # C22
    col += 1
    
    # M2 = (A21+A22)B11
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[1, col] += 1 # C21
    W[3, col] -= 1 # C22 (C22 = M1 - M2 + ...)
    col += 1
    
    # M3 = A11(B12-B22)
    U[0, col] += 1
    V[1, col] += 1; V[6, col] -= 1
    W[2, col] += 1 # C12
    W[3, col] += 1 # C22
    col += 1
    
    # M4 = A22(B21-B11)
    U[4, col] += 1
    V[5, col] += 1; V[0, col] -= 1
    W[0, col] += 1 # C11
    W[1, col] += 1 # C21
    col += 1
    
    # M5 = (A11+A12)B22
    U[0, col] += 1; U[1, col] += 1
    V[6, col] += 1
    W[0, col] -= 1 # C11
    W[2, col] += 1 # C12
    col += 1
    
    # M6 = (A21-A11)(B11+B12)
    U[3, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[3, col] += 1 # C22
    col += 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, col] += 1; U[4, col] -= 1
    V[5, col] += 1; V[6, col] += 1
    W[0, col] += 1 # C11
    col += 1
    
    # Part 2: A_mid B2 (2x2 * 2x3)
    # B2 columns k=2,3,4 (indices 2,3,4 in B)
    # V indices for B2:
    # Row 0 (j=0): k=2->2, k=3->3, k=4->4
    # Row 1 (j=1): k=2->7, k=3->8, k=4->9
    
    # For each column c in 2,3,4:
    # We need 4 terms for 2x2 * 2x1.
    # Let Bc = [B1c, B2c]^T.
    # C1c = A11 B1c + A12 B2c
    # C2c = A21 B1c + A22 B2c
    
    # We can use the following 4-term decomposition for 2x2x1:
    # N1 = A11 B1c
    # N2 = A12 B2c
    # N3 = A21 B1c
    # N4 = A22 B2c
    # This is schoolbook. Rank 4.
    
    for c in range(2, 5):
        # N1
        U[0, col] += 1
        V[c, col] += 1 # B1c is j=0, k=c -> idx c
        W[c*2, col] += 1 # C1c
        col += 1
        
        # N2
        U[1, col] += 1
        V[5+c, col] += 1 # B2c is j=1, k=c -> idx 5+c
        W[c*2, col] += 1 # C1c
        col += 1
        
        # N3
        U[3, col] += 1
        V[c, col] += 1 # B1c
        W[c*2+1, col] += 1 # C2c
        col += 1
        
        # N4
        U[4, col] += 1
        V[5+c, col] += 1 # B2c
        W[c*2+1, col] += 1 # C2c
        col += 1
        
    # Part 3: A_col2 B_row2 (2x1 * 1x5)
    # A_col2: A13 (idx 2), A23 (idx 5)
    # B_row2: B31..B35 (j=2, k=0..4 -> idx 10..14)
    
    # C1k += A13 B3k
    # C2k += A23 B3k
    
    for k in range(5):
        # N1
        U[2, col] += 1
        V[10+k, col] += 1
        W[k*2, col] += 1 # C1k
        col += 1
        
        # N2
        U[5, col] += 1
        V[10+k, col] += 1
        W[k*2+1, col] += 1 # C2k
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
