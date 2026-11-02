"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6

    # Block decomposition: split m=5 into m1=1, m2=4.
    # Part 1: 2x1x6 direct outer product, rank 12.
    # Part 2: 2x4x6, split j into two 2-blocks, k into three 2-blocks,
    #         each 2x2x2 sub-problem uses Strassen (rank 7).
    #         Total Part 2: 2 * 3 * 7 = 42.
    # Grand total rank: 54.

    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Part 1: j=0, direct terms (outer product of A[:,0] and B[0,:])
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Strassen rank-1 tensors for 2x2x2 multiplication.
    # Each entry: (u_comps, v_comps, w_comps)
    #   u_comps: [(local_i, local_j, coeff)] for A[local_i, local_j]
    #   v_comps: [(local_j, local_k, coeff)] for B[local_j, local_k]
    #   w_comps: [(local_i, local_k, coeff)] for C[local_i, local_k]
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11); C00+=M1, C11+=M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2 = (A10+A11)B00; C10+=M2, C11-=M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3 = A00(B01-B11); C01+=M3, C11+=M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4 = A11(B10-B00); C00+=M4, C10+=M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5 = (A00+A01)B11; C00-=M5, C01+=M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6 = (A10-A00)(B00+B01); C11+=M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11); C00+=M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

    # Part 2: j in {1,2,3,4}, split into two j-blocks of size 2
    for block_j in range(2):
        j_base = 1 + block_j * 2
        for block_k in range(3):
            k_base = block_k * 2
            for u_comps, v_comps, w_comps in strassen_terms:
                for (li, lj, val) in u_comps:
                    U[li * m + (j_base + lj)][col] += val
                for (lj, lk, val) in v_comps:
                    V[(j_base + lj) * p + (k_base + lk)][col] += val
                for (li, lk, val) in w_comps:
                    W[(k_base + lk) * n + li][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
