"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split C = A * B (3x3 * 3x5) into C1 = A * B1 (3x3 * 3x3) and C2 = A * B2 (3x3 * 3x2)
    
    # Part 1: Rank 23 algorithm for 3x3 * 3x3
    rank_1 = 23
    # Part 2: Rank 18 (Schoolbook) for 3x3 * 3x2
    rank_2 = 18
    
    total_rank = rank_1 + rank_2
    
    U = [[0] * total_rank for _ in range(n * m)] # 9 x 41
    V = [[0] * total_rank for _ in range(m * p)] # 15 x 41
    W = [[0] * total_rank for _ in range(p * n)] # 15 x 41
    
    # --- Part 1: Rank 23 for 3x3 * 3x3 ---
    # Coefficients for Rank 23 algorithm (3x3x3)
    # U, V, W are 9x23 matrices
    # Using a known Rank 23 decomposition
    
    # Term 1: A11 * B11
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 2: A12 * B21
    U[1][1] = 1; V[3][1] = 1; W[1][1] = 1
    # Term 3: A21 * B12
    U[3][2] = 1; V[1][2] = 1; W[2][2] = 1
    # Term 4: A22 * B22
    U[4][3] = 1; V[4][3] = 1; W[3][3] = 1
    # Term 5: A31 * B13
    U[6][4] = 1; V[2][4] = 1; W[4][4] = 1
    # Term 6: A32 * B23
    U[7][5] = 1; V[5][5] = 1; W[5][5] = 1
    # Term 7: A13 * B31
    U[2][6] = 1; V[6][6] = 1; W[6][6] = 1
    # Term 8: A23 * B32
    U[5][7] = 1; V[7][7] = 1; W[7][7] = 1
    # Term 9: A33 * B33
    U[8][8] = 1; V[8][8] = 1; W[8][8] = 1
    
    # The remaining 14 terms of the Rank 23 algorithm are complex.
    # To ensure validity, I will fallback to Schoolbook for Part 1 if I cannot guarantee the rest.
    # However, I will implement the full Rank 23 algorithm using a standard set.
    
    # Since I cannot guarantee the full Rank 23 coefficients in this environment,
    # I will use Schoolbook (27) for Part 1 and Schoolbook (18) for Part 2.
    # Total Rank 45.
    
    # Wait, I can use Rank 27 for Part 1 and Rank 15 for Part 2?
    # Rank 15 for 3x3x2 is known.
    
    # Let's use Rank 27 + Rank 18 = 45.
    
    # I will implement Rank 27 + Rank 18.
    
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
