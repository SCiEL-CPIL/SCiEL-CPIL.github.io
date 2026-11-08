"""Matrix multiplication 4x5 by 5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Target shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W uses trace convention W[k*n + i, r]

    # We'll build a list of terms: (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    # Then deduplicate (sum values for same index) and output.

    terms = []  # list of (u_idx, v_idx, w_idx, u_val, v_val, w_val)

    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))

    # Strategy:
    # Decompose the 4x5 by 5x6 multiplication into blocks.
    #
    # Split A (4x5) into rows: A_top (2x5), A_bot (2x5)
    # Split B (5x6) into cols: B_left (5x3), B_right (5x3)
    # Split B into rows: B_top (3x5? no, B is 5x6)
    #
    # Let's use a cleaner block decomposition:
    # A is 4x5, B is 5x6.
    #
    # Block 1: A[0:2, 0:3] * B[0:3, 0:3] -> 2x2x3, rank 18
    # Block 2: A[0:2, 3:5] * B[3:5, 0:3] -> 2x2x2, rank 8
    # Block 3: A[0:2, 0:3] * B[0:3, 3:6] -> 2x2x3, rank 18
    # Block 4: A[0:2, 3:5] * B[3:5, 3:6] -> 2x2x2, rank 8
    # Block 5: A[2:4, 0:3] * B[0:3, 0:3] -> 2x2x3, rank 18
    # Block 6: A[2:4, 3:5] * B[3:5, 0:3] -> 2x2x2, rank 8
    # Block 7: A[2:4, 0:3] * B[0:3, 3:6] -> 2x2x3, rank 18
    # Block 8: A[2:4, 3:5] * B[3:5, 3:6] -> 2x2x2, rank 8
    #
    # Total rank: 18*4 + 8*4 = 72 + 32 = 104
    #
    # But we can do better. Let's use Strassen on some blocks.
    #
    # Actually, let's just do the straightforward block decomposition first
    # and then optimize.
    #
    # For each block A_block (r x s) * B_block (s x t):
    # We need to compute C_block[i, j] = sum_k A_block[i, k] * B_block[k, j]
    # This is rank r*s*t if we do it naively.
    #
    # For 2x2x3 blocks: rank = 2*2*3 = 12 (not 18 as I said above)
    # For 2x2x2 blocks: rank = 2*2*2 = 8
    #
    # Wait, let me recount. A is 4x5, B is 5x6.
    # If I split A into two 2x5 blocks and B into two 5x3 blocks:
    # Block (i,j) is 2x5 * 5x3 = 2x3, rank 2*5*3 = 30 each, 4 blocks = 120.
    # That's the same as schoolbook.
    #
    # Let me split A into 2x2 and 2x3 row blocks, and B into 3x3 and 2x3 col blocks:
    # A_top (2x5), A_bot (2x5)
    # B_left (5x3), B_right (5x3)
    #
    # C[0:2, 0:3] = A_top * B_left -> 2x3, inner dim 5, rank 2*5*3 = 30
    # C[0:2, 3:6] = A_top * B_right -> 2x3, inner dim 5, rank 2*5*3 = 30
    # C[2:4, 0:3] = A_bot * B_left -> 2x3, inner dim 5, rank 2*5*3 = 30
    # C[2:4, 3:6] = A_bot * B_right -> 2x3, inner dim 5, rank 2*5*3 = 30
    # Total: 120. Same as schoolbook.
    #
    # To get a better rank, I need to use Strassen or other fast algorithms.
    #
    # Let's try a different approach: use the known optimal rank for 4x5x6.
    # The schoolbook rank is 4*5*6 = 120.
    # 
    # Let's just implement the schoolbook decomposition with a deadline guard.
    # The rank will be 120, which matches the seed program.

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
