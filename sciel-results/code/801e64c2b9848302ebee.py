import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Rank 48 construction for M_{2,4,8}:
    # Split into 4 blocks of 2x2x2 using Strassen (rank 7 each) = 28
    # Plus 4 additional blocks to handle cross terms = 4*7 = 28? No, that's 56.
    # 
    # Alternative: Use a known rank 48 decomposition.
    # Actually, the rank of M_{2,4,8} is known to be 48.
    # Construction: 
    #   - 2 blocks of M_{2,4,4} with rank 24 each? No.
    #   - Use the fact that 4*7 = 28 for the diagonal blocks, and 20 for the off-diagonal.
    #
    # Let's use a different approach: 
    #   The tensor M_{2,4,8} can be decomposed as:
    #   - 4 copies of M_{2,2,2} (rank 7 each) = 28
    #   - Plus additional terms for the remaining structure.
    #
    # Actually, let me use a verified rank 48 construction.
    # The key insight is to use a "block Strassen" approach where we handle
    # the 2x4x8 matrix as a combination of smaller blocks.
    #
    # For now, I'll implement the rank 56 solution but optimize it.
    # Actually, let me try to find a rank 48 solution by using a smarter decomposition.
    #
    # Known result: rank(M_{2,4,8}) = 48.
    # Construction: 
    #   Split A (2x4) into A1 (2x2) and A2 (2x2)
    #   Split B (4x8) into B1 (2x8) and B2 (2x8)
    #   C = A1*B1 + A2*B2
    #   Each A_i*B_i is 2x2*2x8 = 2x2x8
    #   rank(2x2x8) = 14 (by Strassen-like methods)
    #   So total rank = 2*14 = 28? No, that's too optimistic.
    #
    # Actually, rank(2x2x8) = 14 is not correct. Let me reconsider.
    # The rank of M_{2,2,2} is 7. For M_{2,2,4}, it's 14? No.
    # 
    # Let me just implement the rank 56 solution cleanly.

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    A_left_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]

    A_right_coeffs = [
        [(0, 2, 1), (1, 3, 1)],
        [(1, 2, 1), (1, 3, 1)],
        [(0, 2, 1)],
        [(1, 3, 1)],
        [(0, 2, 1), (0, 3, 1)],
        [(1, 2, 1), (0, 2, -1)],
        [(0, 3, 1), (1, 3, -1)]
    ]

    B_top_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]

    B_bot_coeffs = [
        [(2, 0, 1), (3, 1, 1)],
        [(2, 0, 1)],
        [(2, 1, 1), (3, 1, -1)],
        [(3, 0, 1), (2, 0, -1)],
        [(3, 1, 1)],
        [(2, 0, 1), (2, 1, 1)],
        [(3, 0, 1), (3, 1, 1)]
    ]

    C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    for t in range(4):
        for s in range(7):
            c_idx = t * 14 + s
            for i, j, coeff in A_left_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            for j, k_local, coeff in B_top_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff

        for s in range(7):
            c_idx = t * 14 + 7 + s
            for i, j, coeff in A_right_coeffs[s]:
                U[i * m + j][c_idx] += coeff
            for j, k_local, coeff in B_bot_coeffs[s]:
                k_global = t * 2 + k_local
                V[j * p + k_global][c_idx] += coeff
            for i, k_local, coeff in C_coeffs[s]:
                k_global = t * 2 + k_local
                W[k_global * n + i][c_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
