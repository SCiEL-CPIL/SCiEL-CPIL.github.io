import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 13 decomposition
    # Split: A = [A1 (2x2), A2 (2x1)], B = [B1 (2x3); B2 (1x3)]
    # C = A1*B1 + A2*B2
    # Rank(A1*B1) = 11 (Strassen for 2x2x2 part + Schoolbook for 2x2x1 part)
    # Rank(A2*B2) = 2
    
    rank = 13
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1) -> C (cols 0,1)
    # This is <2, 2, 2> multiplication, rank 7 (Strassen)
    # Indices in A: 0,1 (row 0), 3,4 (row 1) for cols 0,1
    # Indices in B: 0,1 (row 0), 3,4 (row 1) for cols 0,1
    # Indices in C: 0,1 (col 0), 2,3 (col 1) for rows 0,1
    
    # Strassen terms for A1*B1 -> C11 (top-left 2x2 of C)
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1 # C10, -C11
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1 # C01, C11
    
    # M4: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1 # C00, C10
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1 # -C00, C01
    
    # M6: (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1 # C00
    
    # Part 2: A1 (cols 0,1) * B2 (col 2 of B, which is row 2? No, B is 3x3. B1 is rows 0,1. B2 is row 2?
    # Wait, B is split into B1 (top 2 rows) and B2 (bottom 1 row).
    # B1 is 2x3. B2 is 1x3.
    # A1 is 2x2. A2 is 2x1.
    # C = A1 B1 + A2 B2.
    # A1 B1: A1 (2x2) * B1 (2x3).
    # B1 has columns 0,1,2.
    # We did Strassen for B1[:, 0:2] (2x2).
    # Now we need A1 * B1[:, 2] (2x1).
    # This is <2, 2, 1>. Rank 4.
    
    # A1 indices: 0,1,3,4.
    # B1[:, 2] indices: 2 (B02), 5 (B12).
    # C[:, 2] indices: 4 (C02), 5 (C12).
    
    # Schoolbook for A1 * B1[:, 2]
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1 # A00 B02 -> C02
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1 # A01 B12 -> C02
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1 # A10 B02 -> C12
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1 # A11 B12 -> C12
    
    # Part 3: A2 (col 2) * B2 (row 2)
    # A2 is 2x1. Indices: 2 (A02), 5 (A12).
    # B2 is 1x3. Indices: 6 (B20), 7 (B21), 8 (B22).
    # C is 2x3.
    # C = A2 * B2.
    # C00 = A02 B20
    # C01 = A02 B21
    # C02 = A02 B22
    # C10 = A12 B20
    # C11 = A12 B21
    # C12 = A12 B22
    
    # Rank 2 decomposition
    # Term 1: A02 * B2 -> C0
    U[2, 11] = 1
    V[6, 11] = 1; V[7, 11] = 1; V[8, 11] = 1
    W[0, 11] = 1; W[2, 11] = 1; W[4, 11] = 1
    
    # Term 2: A12 * B2 -> C1
    U[5, 12] = 1
    V[6, 12] = 1; V[7, 12] = 1; V[8, 12] = 1
    W[1, 12] = 1; W[3, 12] = 1; W[5, 12] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
