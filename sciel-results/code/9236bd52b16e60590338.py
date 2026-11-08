"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Use Strassen-like decomposition for rectangular matrices
    # Pad A (3x3) to 4x4 and B (3x6) to 4x6
    # Then use 7 Strassen products, each of which is a 4x4 * 4x6 product
    
    # Actually, let's think about this more carefully.
    # For 3x3 * 3x6, we can pad to 4x4 * 4x6.
    # The rank of 4x4 * 4x6 matrix multiplication...
    
    # Let me try a different approach: use the known result that 
    # 3x3 * 3x6 can be done with rank 18 using a specific decomposition.
    
    # Actually, let me just use the schoolbook method but verify it's correct.
    # The schoolbook method has rank = n*m*p = 3*3*6 = 54.
    
    # Let me try to find a better decomposition.
    # One approach: group the columns of B.
    # B is 3x6, so we can view it as three 3x2 blocks: B = [B1, B2, B3]
    # Then C = A*B = [A*B1, A*B2, A*B3]
    # Each A*Bj is 3x3 * 3x2, which has rank 3*3*2 = 18 in schoolbook.
    # Total rank = 3 * 18 = 54. Same as before.
    
    # Another approach: use Strassen for the 3x3 part.
    # Pad A to 4x4, and for each column group...
    
    # Let me just implement the schoolbook method correctly and return it.
    # The key is to get the indexing right.
    
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 18 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i][k] += A[i][j] * B[j][k]
                # U[i*m + j][col] = 1
                # V[j*p + k][col] = 1
                # W[k*n + i][col] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
