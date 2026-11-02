"""Refined program for matrix multiplication 2x5x6 using 2+3 split."""
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Split m=5 into m1=2 and m2=3.
    # Part 1: 2x2x6. Split p=6 into three 2-blocks.
    #         Each 2x2x2 block uses Strassen (rank 7).
    #         Total Part 1: 3 * 7 = 21.
    # Part 2: 2x3x6. Split p=6 into three 2-blocks.
    #         Each 2x3x2 block is decomposed into:
    #           - 2x1x2 part (direct, rank 4)
    #           - 2x2x2 part (Strassen, rank 7)
    #           Total per block: 11.
    #         Total Part 2: 3 * 11 = 33.
    # Grand total rank: 21 + 33 = 54.

    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Strassen rank-1 tensors for 2x2x2 multiplication.
    # Indices: (i, j) for A, (j, k) for B, (i, k) for C
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    # Helper to add a 2x2x2 Strassen block
    # i_base: starting row index for A (0..1 offset)
    # j_base: starting col index for A / row index for B
    # k_base: starting col index for B / row index for C
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        for u_comps, v_comps, w_comps in strassen_terms:
            for (li, lj, val) in u_comps:
                U[(i_base + li) * m + (j_base + lj)][col] += val
            for (lj, lk, val) in v_comps:
                V[(j_base + lj) * p + (k_base + lk)][col] += val
            for (li, lk, val) in w_comps:
                W[(k_base + lk) * n + (i_base + li)][col] += val
            col += 1

    # Helper to add a 2x1x2 direct block
    # A is 2x1 (col j), B is 1x2 (row j)
    # Result C is 2x2
    def add_direct_212(i_base, j_idx, k_base):
        nonlocal col
        # A[0, j] * B[j, 0] -> C[0, 0]
        U[i_base * m + j_idx][col] = 1
        V[j_idx * p + k_base][col] = 1
        W[k_base * n + i_base][col] = 1
        col += 1
        
        # A[0, j] * B[j, 1] -> C[0, 1]
        U[i_base * m + j_idx][col] = 1
        V[j_idx * p + (k_base + 1)][col] = 1
        W[(k_base + 1) * n + i_base][col] = 1
        col += 1
        
        # A[1, j] * B[j, 0] -> C[1, 0]
        U[(i_base + 1) * m + j_idx][col] = 1
        V[j_idx * p + k_base][col] = 1
        W[k_base * n + (i_base + 1)][col] = 1
        col += 1
        
        # A[1, j] * B[j, 1] -> C[1, 1]
        U[(i_base + 1) * m + j_idx][col] = 1
        V[j_idx * p + (k_base + 1)][col] = 1
        W[(k_base + 1) * n + (i_base + 1)][col] = 1
        col += 1

    # Part 1: j in {0, 1} (2x2x6)
    # Split k into 3 blocks of 2
    for block_k in range(3):
        k_base = block_k * 2
        add_strassen_block(0, 0, k_base)

    # Part 2: j in {2, 3, 4} (2x3x6)
    # Split k into 3 blocks of 2
    for block_k in range(3):
        k_base = block_k * 2
        # For each 2x3x2 block:
        # Decompose into 2x1x2 (j=2) + 2x2x2 (j=3,4)
        # 1. Direct 2x1x2 for j=2
        add_direct_212(0, 2, k_base)
        # 2. Strassen 2x2x2 for j=3,4
        add_strassen_block(0, 3, k_base)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
