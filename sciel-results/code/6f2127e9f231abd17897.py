"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 factors
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],    # b
        [0, 1, 0, 0, 0, 1, 0],    # c
        [1, 1, 0, 1, 0, 0, -1]    # d
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 1, 0],    # f
        [0, 0, 0, 1, 0, 0, 1],    # g
        [1, 0, -1, 0, 1, 0, 1]    # h
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]

    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # This creates 2x2x2 Strassen blocks at:
    #   (n=1, m=2, p=2) -> indices i=[0], j=[0,1], k=[0,1]
    #   (n=2, m=2, p=2) -> indices i=[1,2], j=[2,3], k=[2,3]
    # Wait, let me think more carefully about the partition.
    # 
    # Partition n into [1, 2]: blocks of size 1 and 2
    # Partition m into [2, 1, 2]: blocks of size 2, 1, 2
    # Partition p into [2, 1, 2]: blocks of size 2, 1, 2
    #
    # Strassen applies to 2x2x2 sub-blocks. We need all three dimensions to be size 2.
    # n-blocks: [1, 2] -> only one block of size 2 (the second one, indices 1,2)
    # m-blocks: [2, 1, 2] -> two blocks of size 2 (indices 0,1 and 3,4)
    # p-blocks: [2, 1, 2] -> two blocks of size 2 (indices 0,1 and 3,4)
    #
    # So we get 1 * 2 * 2 = 4 Strassen 2x2x2 blocks:
    #   Block 1: i=[1,2], j=[0,1], k=[0,1]
    #   Block 2: i=[1,2], j=[0,1], k=[3,4]
    #   Block 3: i=[1,2], j=[3,4], k=[0,1]
    #   Block 4: i=[1,2], j=[3,4], k=[3,4]
    #
    # Each Strassen block: rank 7 instead of 8 (saves 1)
    # 4 blocks save 4 total: 75 - 4 = 71

    # Define the Strassen blocks
    strassen_blocks = [
        ([1, 2], [0, 1], [0, 1]),
        ([1, 2], [0, 1], [3, 4]),
        ([1, 2], [3, 4], [0, 1]),
        ([1, 2], [3, 4], [3, 4]),
    ]

    # Build set of (i,j,k) tuples covered by Strassen blocks
    covered = set()
    for i_list, j_list, k_list in strassen_blocks:
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    covered.add((i, j, k))

    # Total rank: 4*7 + (75 - 4*8) = 28 + 43 = 71
    rank_total = 4 * 7 + (n * m * p - 4 * 8)
    assert rank_total == 71, f"Expected 71, got {rank_total}"

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0
    for i_list, j_list, k_list in strassen_blocks:
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list

        # Map local indices to global
        u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]  # a, b, c, d
        v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]  # e, f, g, h
        w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]  # C11, C12, C21, C22

        for c in range(7):
            for s in range(4):
                U[u_rows[s]][col + c] = U_strassen[s][c]
                V[v_rows[s]][col + c] = V_strassen[s][c]
                W[w_rows[s]][col + c] = W_strassen[s][c]
        col += 7

    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    assert col == rank_total, f"Expected rank {rank_total}, got {col}"

    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
