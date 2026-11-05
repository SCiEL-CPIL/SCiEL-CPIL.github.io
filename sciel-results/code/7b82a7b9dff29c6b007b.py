"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen factors for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
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

    # Define 4 independent 2x2x2 Strassen blocks
    # n=3: use i in {0,1} for all blocks
    # m=5: use j in {0,1} and j in {2,3}
    # p=5: use k in {0,1} and k in {2,3}
    blocks = [
        (i_idx, j_idx, k_idx)
        for i_idx in [[0, 1]]
        for j_idx in [[0, 1], [2, 3]]
        for k_idx in [[0, 1], [2, 3]]
    ]

    # Mark which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for i_idx, j_idx, k_idx in blocks:
        for i in i_idx:
            for j in j_idx:
                for k in k_idx:
                    covered.add((i, j, k))

    # Count rank: 4 blocks * 7 + remaining schoolbook terms
    num_schoolbook = n * m * p - len(covered)
    rank_total = len(blocks) * 7 + num_schoolbook

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0
    for i_idx, j_idx, k_idx in blocks:
        i0, i1 = i_idx
        j0, j1 = j_idx
        k0, k1 = k_idx

        # Map local indices to global rows
        u_rows = [i0*m+j0, i0*m+j1, i1*m+j0, i1*m+j1]  # a,b,c,d
        v_rows = [j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1]  # e,f,g,h
        w_rows = [k0*n+i0, k1*n+i0, k0*n+i1, k1*n+i1]  # C11,C12,C21,C22

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

    # Verify correctness using numpy for speed
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat, dtype=np.float64)
        V = np.array(V_mat, dtype=np.float64)
        W = np.array(W_mat, dtype=np.float64)
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Reshape and use einsum-like approach
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-10:
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
