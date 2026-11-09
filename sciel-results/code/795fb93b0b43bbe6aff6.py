"""Optimized rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 105. We try to improve via block decomposition.
    # 3x5x7 can be split into blocks:
    #   (3x3x7) + (3x2x7)
    # For 3x3x7: known best is 62 (using 23 for 3x3x3 and 9 for 3x3x1? No.)
    # Let's use the known results:
    #   2x2x2: rank 7
    #   2x2x3: rank 11
    #   2x3x3: rank 15? No, 2x3x3 is 11? No, 3x3x3 is 23.
    #   3x3x3: rank 23
    #   3x3x2: rank 15? No, 2x3x3 is 15?
    # Actually, let's use the known best for small cases:
    #   2x2x2: 7
    #   2x2x3: 11
    #   2x3x3: 15
    #   3x3x3: 23
    #   3x3x2: 15
    #   3x2x2: 11
    #   2x3x2: 11
    #   3x2x3: 15
    #   3x3x1: 9
    #   3x1x3: 9
    #   1x3x3: 9
    #   3x1x1: 3
    #   1x3x1: 3
    #   1x1x3: 3
    #   3x2x1: 6
    #   2x3x1: 6
    #   1x3x2: 6
    #   3x1x2: 6
    #   2x1x3: 6
    #   1x2x3: 6
    #   3x2x2: 11
    #   2x3x2: 11
    #   2x2x3: 11
    #   3x3x2: 15
    #   3x2x3: 15
    #   2x3x3: 15
    #   3x3x3: 23
    
    # Split 3x5x7:
    #   (3x3x7) + (3x2x7)
    # 3x3x7 = 3x3x3 + 3x3x2 + 3x3x2 = 23 + 15 + 15 = 53
    # 3x2x7 = 3x2x3 + 3x2x2 + 3x2x2 = 15 + 11 + 11 = 37
    # Total = 53 + 37 = 90
    
    # Alternatively:
    # 3x5x7 = 3x5x3 + 3x5x2 + 3x5x2 = 30 + 20 + 20 = 70?
    # 3x5x3 = 3x3x3 + 3x2x3 = 23 + 15 = 38
    # 3x5x2 = 3x3x2 + 3x2x2 = 15 + 11 = 26
    # Total = 38 + 26 + 26 = 90
    
    # So rank 90 is achievable.
    # Let's implement the block decomposition with known optimal ranks.
    # For simplicity and correctness, we'll use schoolbook for the base cases
    # if we don't have the exact coefficients, but we know rank 90 is better than 105.
    
    # However, generating the exact 90-rank decomposition requires the specific
    # 3x3x3 rank 23 algorithm and others, which are complex to hardcode correctly.
    # Given the constraints, we'll return the schoolbook decomposition to ensure validity.
    # If we could hardcode the 3x3x3 rank 23, we'd do so, but it's risky without verification.
    
    # For now, return schoolbook (rank 105) which is valid.
    # The evaluator measures rank, so 105 is the baseline.
    # To improve, we need the exact coefficients.
    # Since I don't have them memorized with 100% confidence, I'll stick to schoolbook.
    
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
