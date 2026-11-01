"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use two rank-7 Strassen-like decompositions for 3x3 x 3x3
    # B is 3x6, split into B_left (3x3) and B_right (3x3)
    # C = A @ B = [A @ B_left | A @ B_right]
    
    # Rank-7 decomposition for 3x3 x 3x3 (one known construction)
    # We'll define U, V, W for a single 3x3 x 3x3 multiplication
    # Then combine two such decompositions
    
    # For 3x3 x 3x3, indices: A[i,j], B[j,k] -> C[i,k]
    # U has shape (n*m, rank) = (9, 7)
    # V has shape (m*p, rank) = (9, 7)  
    # W has shape (p*n, rank) = (9, 7)
    
    # Known rank-7 decomposition coefficients for 3x3 x 3x3
    # Using a standard construction
    U_single = [
        [1, 0, 0, 0, 0, 0, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 0],  # A[0,1]
        [0, 0, 0, 0, 0, 0, 0],  # A[0,2] - placeholder, need proper values
        [0, 1, 0, 0, 0, 0, 0],  # A[1,0]
        [0, 0, 0, 0, 0, 0, 0],  # A[1,1]
        [0, 0, 0, 0, 0, 0, 0],  # A[1,2]
        [0, 0, 1, 0, 0, 0, 0],  # A[2,0]
        [0, 0, 0, 0, 0, 0, 0],  # A[2,1]
        [0, 0, 0, 0, 0, 0, 0],  # A[2,2]
    ]
    
    # This is getting complicated. Let me use a different approach.
    # I'll use the schoolbook but try to find a lower rank via simple optimization.
    # For now, return schoolbook rank 54, but let me try rank 18.
    
    # Actually, let me just return the schoolbook for safety, since I'm not sure
    # of the exact rank-7 coefficients.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
