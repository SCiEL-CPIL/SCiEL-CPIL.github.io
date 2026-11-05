"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using Strassen's algorithm for two 3x3 multiplications.
    n, m, p = DIMENSIONS
    rank = 14
    
    # Strassen factors for 3x3 matrix multiplication
    # u vectors for A (length 9)
    # v vectors for B (length 9)
    # w vectors for C (length 9)
    
    # Indices: 0..8 correspond to A11, A12, A13, A21, A22, A23, A31, A32, A33
    
    u_factors = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1],  # M1: A11 + A33
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  # M2: A21 + A22
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # M3: A11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # M4: A33
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # M5: A11 + A12
        [-1, 0, 0, 0, 0, 0, 1, 0, 0], # M6: A31 - A11
        [0, 0, 0, 0, 1, 0, 0, 0, -1]  # M7: A22 - A33
    ]
    
    v_factors = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1],  # M1: B11 + B33
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # M2: B11
        [0, 1, 0, 0, 0, 0, 0, -1, 0], # M3: B12 - B32
        [-1, 0, 0, 1, 0, 0, 0, 0, 0], # M4: B21 - B11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # M5: B33
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # M6: B11 + B12
        [0, 0, 0, 1, 1, 0, 0, 0, 0]   # M7: B21 + B22
    ]
    
    w_factors = [
        [1, 0, 1, 0, 0, 0, 1, 0, -1], # M1: C11+=, C13+=, C31+=, C33-=
        [0, 0, 0, 1, 1, 0, 0, 0, 0],  # M2: C21+=, C22+=
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # M3: C12+=, C13+=
        [0, 0, 0, 0, 0, 0, 1, 1, 0],  # M4: C31+=, C32+=
        [0, 0, 1, 0, 0, 1, 0, 0, 0],  # M5: C13+=, C23+=
        [0, 0, 0, 0, 0, 0, -1, -1, 0],# M6: C31-=, C32-=
        [0, 0, 0, -1, -1, 0, 0, 0, 0] # M7: C21-=, C22-=
    ]
    
    # Initialize matrices
    # U: (n*m, rank) = (9, 14)
    # V: (m*p, rank) = (18, 14)
    # W: (p*n, rank) = (18, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U, V, W for the first block (columns 0-6)
    # This block computes C1 = A * B1
    # B1 is the first 3 columns of B, so rows 0-8 of V.
    # C1 is the first 3 columns of C, so rows 0-8 of W.
    
    for k in range(7):
        # U part
        for i in range(9):
            U[i][k] = u_factors[k][i]
            
        # V part (rows 0-8 for B1)
        for j in range(9):
            V[j][k] = v_factors[k][j]
        # Rows 9-18 for B2 are 0 (already initialized)
        
        # W part (rows 0-8 for C1)
        for c_idx in range(9):
            W[c_idx][k] = w_factors[k][c_idx]
        # Rows 9-18 for C2 are 0 (already initialized)
        
    # Fill U, V, W for the second block (columns 7-13)
    # This block computes C2 = A * B2
    # B2 is the last 3 columns of B, so rows 9-17 of V.
    # C2 is the last 3 columns of C, so rows 9-17 of W.
    
    for k in range(7):
        col_idx = 7 + k
        
        # U part (same as before, A is shared)
        for i in range(9):
            U[i][col_idx] = u_factors[k][i]
            
        # V part (rows 9-17 for B2)
        for j in range(9):
            V[9 + j][col_idx] = v_factors[k][j]
        # Rows 0-9 for B1 are 0 (already initialized)
        
        # W part (rows 9-17 for C2)
        for c_idx in range(9):
            W[9 + c_idx][col_idx] = w_factors[k][c_idx]
        # Rows 0-9 for C1 are 0 (already initialized)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
