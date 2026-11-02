"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x1x2 blocks
    # Split p=10 into 5 blocks of width 2
    num_blocks = p // 2
    rank = 11 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(num_blocks):
        # Block columns in B are 2k and 2k+1
        c1 = 2 * k
        c2 = 2 * k + 1
        
        # --- Strassen Part (Rank 7) ---
        # M1 = (A00 + A11)(B0c1 + B1c2)
        U[0][col] += 1
        U[3][col] += 1  # A10 is index 3? No. U index i*m+j. A[0,0]->0, A[1,1]->4.
        # Wait, U indices:
        # A[0,0] -> 0*3+0 = 0
        # A[1,1] -> 1*3+1 = 4
        # Let's re-verify U indices.
        # n=2, m=3.
        # A[0,0]=0, A[0,1]=1, A[0,2]=2
        # A[1,0]=3, A[1,1]=4, A[1,2]=5
        
        # Correcting U indices for Strassen:
        # M1: A00, A11 -> U[0], U[4]
        U[0][col] = 1
        U[4][col] = 1
        V[0 * p + c1][col] = 1
        V[1 * p + c2][col] = 1
        W[c1 * n + 0][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11)B0c1
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        W[c2 * n + 1][col] = -1
        col += 1
        
        # M3 = A00(B1c1 - B1c2)
        U[0][col] = 1
        V[1 * p + c1][col] = 1
        V[1 * p + c2][col] = -1
        W[c2 * n + 0][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M4 = A11(B0c2 - B0c1)
        U[4][col] = 1
        V[0 * p + c2][col] = 1
        V[0 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01)B1c2
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + c2][col] = 1
        W[c1 * n + 0][col] = -1
        W[c2 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B0c1 + B0c2)
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + c1][col] = 1
        V[0 * p + c2][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B1c1 + B1c2)
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + c1][col] = 1
        V[1 * p + c2][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # --- Schoolbook Part (Rank 4) ---
        # N1 = A02 B2c1
        U[2][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # N2 = A02 B2c2
        U[2][col] = 1
        V[2 * p + c2][col] = 1
        W[c2 * n + 0][col] = 1
        col += 1
        
        # N3 = A12 B2c1
        U[5][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # N4 = A12 B2c2
        U[5][col] = 1
        V[2 * p + c2][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
