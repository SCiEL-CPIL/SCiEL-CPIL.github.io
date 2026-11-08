"""Strassen-extended tensor decomposition for 2x3 * 3x8 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Rank-minimal exact decomposition: Strassen for (2,2,8) + schoolbook for (2,1,8)
    # Strassen 2x2x2 factors (each row is a coefficient vector of length 4, indexed by (row, col) flattened)
    S_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    S_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    S_W = [
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]

    # Verify Strassen correctness internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        # Fallback: full schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the (2,3,8) decomposition
    # 4 blocks of Strassen (2x2x2) for the 2x2x8 part: 4 * 7 = 28
    # Schoolbook for the 2x1x8 part: 2 * 1 * 8 = 16
    # Total rank = 28 + 16 = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), rank 28 via 4 Strassen blocks on k-dimension
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # U: indices for (i, a) where i in {0,1}, a in {0,1}
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U[li * m + la][col] = u_val

            # V: indices for (a, k) where a in {0,1}, k in {k0, k0+1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V[la * p + (k0 + lk)][col] = v_val

            # W: indices for (k, i) where k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W[(k0 + lk) * n + li][col] = w_val

            col += 1
            if not _elapsed_ok():
                # Return best partial result (schoolbook) if time runs out
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                c = 0
                for i in range(n):
                    for a in range(m):
                        for k in range(p):
                            U_fb[i * m + a][c] = 1
                            V_fb[a * p + k][c] = 1
                            W_fb[k * n + i][c] = 1
                            c += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Part 2: A2 (2x1) * B2 (1x8), rank 16 schoolbook
    # C[i, k] = A[i, 2] * B[2, k]
    # Tensor condition: sum_r U[i',2,r] V[2,k'',r] W[k'',i',r] = delta[i',i] delta[k'',k]
    # This requires rank 16 (one for each i,k pair)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
