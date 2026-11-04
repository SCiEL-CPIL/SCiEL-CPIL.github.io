"""Matrix multiplication 3x3 by 3x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: split p=5 into 3+2
    # Part 1: 3x3 * 3x3 (trivial rank 27)
    # Part 2: 3x3 * 3x2 (trivial rank 18)
    # Total: 45 (same as full trivial, but structured)
    
    # Try to use a known rank-reduced algorithm for the 3x3*3x3 part
    # For now, use trivial decomposition for both parts
    rank1 = n * m * 3  # 27 for 3x3*3x3
    rank2 = n * m * 2  # 18 for 3x3*3x2
    rank = rank1 + rank2  # 45
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: columns 0-2 of B (3x3 block)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: columns 3-4 of B (3x2 block)
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
