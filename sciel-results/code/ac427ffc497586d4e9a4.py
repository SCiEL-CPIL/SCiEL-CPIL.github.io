"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p into two blocks of size p1 and p2
    p1 = p // 2
    p2 = p - p1
    
    # The problem is decomposed into two sub-problems:
    # 1. Multiply 2x3 by 3xp1
    # 2. Multiply 2x3 by 3xp2
    # The total rank is the sum of the ranks of the sub-problems.
    # We use the schoolbook rank for the sub-problems as a baseline.
    # Rank for sub-problem 1: n * m * p1
    # Rank for sub-problem 2: n * m * p2
    # Total rank: n * m * (p1 + p2) = n * m * p
    
    rank1 = n * m * p1
    rank2 = n * m * p2
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct for Block 1 (k in 0..p1-1)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Construct for Block 2 (k in p1..p-1)
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                # Note: W index is k*n + i. 
                # Since k >= p1, these indices are distinct from Block 1's W indices.
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
