"""Matrix multiplication 4x5x6 with rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 4, 5, 6

    # Strassen's 7-term decomposition for 2x2x2 matrix multiplication
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M5 + M6
    # C[1,0] = M2 + M3
    # C[1,1] = M1 - M3 + M4 + M7
    #
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0] - B[1,1])
    # M4 = A[1,1] * (B[0,1] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * B[0,0]
    # M7 = (A[1,1] - A[0,1]) * (B[0,1] + B[1,1])
    #
    # U factors (for A, indexed by (i,j)):
    strassen_U = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(1, 0, 1), (1, 1, 1)],
        2: [(0, 0, 1)],
        3: [(1, 1, 1)],
        4: [(0, 0, 1), (0, 1, 1)],
        5: [(1, 0, 1), (0, 0, -1)],
        6: [(1, 1, 1), (0, 1, -1)],
    }

    # V factors (for B, indexed by (j,k)):
    strassen_V = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(0, 0, 1)],
        2: [(1, 0, 1), (1, 1, -1)],
        3: [(0, 1, 1), (0, 0, -1)],
        4: [(1, 1, 1)],
        5: [(0, 0, 1)],
        6: [(0, 1, 1), (1, 1, 1)],
    }

    # W factors (for C, indexed by (k,i)):
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M5 + M6
    # C[1,0] = M2 + M3
    # C[1,1] = M1 - M3 + M4 + M7
    strassen_W = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, -1)],
        (0, 1): [(4, 1), (5, 1)],
        (1, 0): [(1, 1), (2, 1)],
        (1, 1): [(0, 1), (2, -1), (3, 1), (6, 1)],
    }

    def make_strassen_block(i_base, j_base, k_base, U, V, W, rank_offset):
        """Add a Strassen 2x2x2 block starting at (i_base, j_base, k_base)."""
        for c in range(7):
            col = rank_offset + c
            # U factor
            for (ii, jj, coeff) in strassen_U[c]:
                U[(i_base + ii) * m + (j_base + jj)][col] += coeff
            # V factor
            for (jj, kk, coeff) in strassen_V[c]:
                V[(j_base + jj) * p + (k_base + kk)][col] += coeff
            # W factor
            for (kk, ii) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                pass  # handled below

        for (kk, ii), factors in strassen_W.items():
            for (c, coeff) in factors:
                col = rank_offset + c
                W[(k_base + kk) * n + (i_base + ii)][col] += coeff

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Use Strassen's algorithm on 2x2 blocks where possible
    # Partition: n=4 into 2+2, m=5 into 2+2+1, p=6 into 2+2+2
    #
    # Blocks:
    # (i:0-1, j:0-1, k:0-1): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:0-1, k:2-3): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:0-1, k:4-5): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:2-3, k:0-1): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:2-3, k:2-3): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:2-3, k:4-5): 2x2x2 -> Strassen rank 7
    # (i:0-1, j:4,   k:0-1): 2x1x2 -> schoolbook rank 4
    # (i:0-1, j:4,   k:2-3): 2x1x2 -> schoolbook rank 4
    # (i:0-1, j:4,   k:4-5): 2x1x2 -> schoolbook rank 4
    # (i:2-3, j:0-1, k:0-1): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:0-1, k:2-3): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:0-1, k:4-5): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:2-3, k:0-1): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:2-3, k:2-3): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:2-3, k:4-5): 2x2x2 -> Strassen rank 7
    # (i:2-3, j:4,   k:0-1): 2x1x2 -> schoolbook rank 4
    # (i:2-3, j:4,   k:2-3): 2x1x2 -> schoolbook rank 4
    # (i:2-3, j:4,   k:4-5): 2x1x2 -> schoolbook rank 4
    #
    # Total: 12 * 7 + 6 * 4 = 84 + 24 = 108

    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    rank_offset = 0

    # 12 Strassen 2x2x2 blocks
    for i_base in [0, 2]:
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                make_strassen_block(i_base, j_base, k_base, U, V, W, rank_offset)
                rank_offset += 7

    # 6 schoolbook 2x1x2 blocks (j=4)
    for i_base in [0, 2]:
        for k_base in [0, 2, 4]:
            for ii in range(2):
                for kk in range(2):
                    col = rank_offset
                    U[(i_base + ii) * m + 4][col] = 1
                    V[4 * p + (k_base + kk)][col] = 1
                    W[(k_base + kk) * n + (i_base + ii)][col] = 1
                    rank_offset += 1

    # Verify the decomposition
    if not verify(U, V, W, rank):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
