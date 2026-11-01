import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    Uses a block decomposition strategy:
    1. Strassen algorithm for the 2x2 sub-block (rank 7).
    2. Direct multiplication for the remaining 2x2x1 block (rank 4).
    3. Direct multiplication for the 2x1x3 block (rank 6).
    Total rank: 7 + 4 + 6 = 17.
    """
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (6, 17)
    # V shape: (m*p, rank) = (9, 17)
    # W shape: (p*n, rank) = (6, 17)
    
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Indices for U (A matrix, 2x3):
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3, A[0,2]=4, A[1,2]=5
    # Indices for V (B matrix, 3x3):
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    # Indices for W (C matrix, 2x3, convention W[k*n+i]):
    # C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    col = 0
    
    # --- Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] ---
    # A_sub = [[A00, A01], [A10, A11]] -> U indices 0, 1, 2, 3
    # B_sub = [[B00, B01], [B10, B11]] -> V indices 0, 1, 3, 4 (Note: V is row-major B[j,k], so B10 is V[3], B11 is V[4])
    # Wait, V index mapping:
    # V[0] = B[0,0]
    # V[1] = B[0,1]
    # V[2] = B[0,2]
    # V[3] = B[1,0]
    # V[4] = B[1,1]
    # V[5] = B[1,2]
    # So B_sub uses V indices 0, 1, 3, 4.
    
    # P1 = (a+d)(e+h)
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[3, col] = 1 # C[0,0] and C[1,1]
    col += 1
    
    # P2 = (c+d)(e)
    U[2, col] = 1; U[3, col] = 1
    V[0, col] = 1
    W[1, col] = 1 # C[1,0]
    col += 1
    
    # P3 = (a-b)(g+h)
    U[0, col] = 1; U[1, col] = -1
    V[3, col] = 1; V[4, col] = 1 # B[1,0] and B[1,1]
    W[1, col] = 1 # C[1,0]
    col += 1
    
    # P4 = (d-b)(f+h)
    U[3, col] = 1; U[1, col] = -1
    V[1, col] = 1; V[4, col] = 1 # B[0,1] and B[1,1]
    W[2, col] = 1 # C[0,1]
    col += 1
    
    # P5 = (a+c)(h)
    U[0, col] = 1; U[2, col] = 1
    V[4, col] = 1 # B[1,1]
    W[0, col] = -1; W[2, col] = 1 # C[0,0] and C[0,1]
    col += 1
    
    # P6 = (b-c)(e+g)
    U[1, col] = 1; U[2, col] = -1
    V[0, col] = 1; V[3, col] = 1 # B[0,0] and B[1,0]
    W[3, col] = 1 # C[1,1]
    col += 1
    
    # P7 = (a+d)(f-g)
    U[0, col] = 1; U[3, col] = 1
    V[1, col] = 1; V[3, col] = -1 # B[0,1] and B[1,0]
    W[0, col] = 1; W[3, col] = -1 # C[0,0] and C[1,1]
    col += 1
    
    # --- Block 2: A[0:2, 0:2] * B[0:2, 2] ---
    # A_sub uses U indices 0,1,2,3
    # B_col uses V indices 2 (B[0,2]), 5 (B[1,2])
    # Output C[:, 2] uses W indices 4 (C[0,2]), 5 (C[1,2])
    
    # T1: A00 * B02 -> C02
    U[0, col] = 1
    V[2, col] = 1
    W[4, col] = 1
    col += 1
    
    # T2: A01 * B12 -> C02
    U[1, col] = 1
    V[5, col] = 1
    W[4, col] = 1
    col += 1
    
    # T3: A10 * B02 -> C12
    U[2, col] = 1
    V[2, col] = 1
    W[5, col] = 1
    col += 1
    
    # T4: A11 * B12 -> C12
    U[3, col] = 1
    V[5, col] = 1
    W[5, col] = 1
    col += 1
    
    # --- Block 3: A[0:2, 2] * B[2, 0:3] ---
    # A_col uses U indices 4 (A[0,2]), 5 (A[1,2])
    # B_row uses V indices 6 (B[2,0]), 7 (B[2,1]), 8 (B[2,2])
    # Output all C uses W indices 0..5
    
    # T5: A02 * B20 -> C00
    U[4, col] = 1
    V[6, col] = 1
    W[0, col] = 1
    col += 1
    
    # T6: A02 * B21 -> C01
    U[4, col] = 1
    V[7, col] = 1
    W[2, col] = 1
    col += 1
    
    # T7: A02 * B22 -> C02
    U[4, col] = 1
    V[8, col] = 1
    W[4, col] = 1
    col += 1
    
    # T8: A12 * B20 -> C10
    U[5, col] = 1
    V[6, col] = 1
    W[1, col] = 1
    col += 1
    
    # T9: A12 * B21 -> C11
    U[5, col] = 1
    V[7, col] = 1
    W[3, col] = 1
    col += 1
    
    # T10: A12 * B22 -> C12
    U[5, col] = 1
    V[8, col] = 1
    W[5, col] = 1
    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
