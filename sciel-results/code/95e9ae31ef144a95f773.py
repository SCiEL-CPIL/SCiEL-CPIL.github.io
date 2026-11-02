"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    # Total rank is 30
    # 18 for the 3x3 part (two 3x3x3 blocks)
    # 12 for the 3x2 part (two 3x2x3 blocks)
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in {0, 1, 2}
    # Sub-part 1a: k in {0, 1, 2}
    for i in range(n):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Sub-part 1b: k in {3, 4, 5}
    for i in range(n):
        for j in range(3):
            for k_prime in range(3):
                k = k_prime + 3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: j in {3, 4}
    # Sub-part 2a: k in {0, 1, 2}
    for i in range(n):
        for j_prime in range(2):
            j = j_prime + 3
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Sub-part 2b: k in {3, 4, 5}
    for i in range(n):
        for j_prime in range(2):
            j = j_prime + 3
            for k_prime in range(3):
                k = k_prime + 3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
