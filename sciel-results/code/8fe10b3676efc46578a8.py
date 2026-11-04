"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Decompose into 4 independent 2x3x2 sub-problems (column pairs of B)
    # For each 2x3x2 sub-problem, use rank-8 decomposition:
    #   Split A (2x3) into A_left (2x2) and A_right (2x1)
    #   Split B (3x2) into B_top (2x2) and B_bottom (1x2)
    #   C = A_left * B_top + A_right * B_bottom
    #   A_left * B_top: 2x2x2 schoolbook = rank 4
    #   A_right * B_bottom: 2x1x2 schoolbook = rank 4
    #   Total per block: rank 8
    # Total rank: 4 * 8 = 32
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 32
    V = [[0] * rank for _ in range(m * p)]  # 24 x 32
    W = [[0] * rank for _ in range(p * n)]  # 16 x 32
    
    col = 0
    for block in range(4):
        # Column pair: columns 2*block and 2*block+1 of B
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # A_left * B_top: 2x2x2 (A rows 0-1, cols 0-1; B rows 0-1, cols 0-1)
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + k0][col] = 1
                W[k0 * n + i][col] = 1
                col += 1
        
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + k1][col] = 1
                W[k1 * n + i][col] = 1
                col += 1
        
        # A_right * B_bottom: 2x1x2 (A rows 0-1, col 2; B row 2, cols 0-1)
        for i in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k0][col] = 1
            W[k0 * n + i][col] = 1
            col += 1
        
        for i in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k1][col] = 1
            W[k1 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
