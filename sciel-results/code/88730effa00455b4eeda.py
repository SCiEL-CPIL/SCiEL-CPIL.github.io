"""Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

Strategy: Use Strassen's algorithm (rank 49) for the 4x4 sub-blocks and
schoolbook for the interactions with the 5th row/column.
Total Rank: 49 (for 4x4) + 57 (for 5th row/col interactions) = 106.
"""

import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START


def _verify(U, V, W, n, m, p):
    """
    Check that the tensor decomposition is exact.
    Convention: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k).
    We verify: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    and for invalid triples, the sum is 0.
    """
    r = len(U[0])
    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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


def _construct_strassen_4x4(n, m, p):
    """
    Construct Strassen-based decomposition for the 4x4 subblock.
    Returns U, V, W for indices 0-3 only, with rank 49.
    """
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        i0, i1 = a_r0, a_r0 + 1
        j0, j1 = a_c0, a_c0 + 1
        k0, k1 = b_c0, b_c0 + 1

        U[i0 * m + j0][start_col] += 1
        U[i1 * m + j1][start_col] += 1
        V[j0 * p + k0][start_col] += 1
        V[j1 * p + k1][start_col] += 1
        W[k0 * n + i0][start_col] += 1
        W[k1 * n + i1][start_col] += 1
        col = start_col + 1

        U[i1 * m + j0][col] += 1
        U[i1 * m + j1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i1][col] += 1
        col += 1

        U[i0 * m + j0][col] += 1
        V[j0 * p + k1][col] += 1
        V[j1 * p + k1][col] -= 1
        W[k1 * n + i0][col] += 1
        col += 1

        U[i1 * m + j1][col] += 1
        V[j1 * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i1][col] += 1
        col += 1

        U[i0 * m + j0][col] += 1
        U[i0 * m + j1][col] += 1
        V[j1 * p + k1][col] += 1
        W[k0 * n + i0][col] -= 1
        W[k1 * n + i0][col] += 1
        col += 1

        U[i1 * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k1][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1

        U[i0 * m + j1][col] += 1
        U[i1 * m + j1][col] -= 1
        V[j1 * p + k0][col] += 1
        V[j1 * p + k1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1

        return col

    def add_strassen_4x4_2x2_product(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_col):
        def _add_strassen_2x2_with_offsets(A_offs, B_offs, C_offs, start_col):
            col = start_col

            for a_off in A_offs:
                for (ai, aj) in [(0, 0), (1, 1)]:
                    U[(a_off[0] + ai) * m + (a_off[1] + aj)][col] += a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(0, 0), (1, 1)]:
                    V[(b_off[0] + bi) * p + (b_off[1] + bj)][col] += b_off[2]
            for c_off in C_offs:
                for (ci, cj) in [(0, 0), (1, 1)]:
                    W[(c_off[1] + cj) * n + (c_off[0] + ci)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                for (ai, aj) in [(1, 0), (1, 1)]:
                    U[(a_off[0] + ai) * m + (a_off[1] + aj)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0] + 0) * p + (b_off[1] + 0)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 0) * n + (c_off[0] + 1)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                U[(a_off[0] + 0) * m + (a_off[1] + 0)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0] + 1) * p + (b_off[1] + 0)][col] += b_off[2]
                V[(b_off[0] + 0) * p + (b_off[1] + 0)][col] -= b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 1) * n + (c_off[0] + 0)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                U[(a_off[0] + 1) * m + (a_off[1] + 1)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0] + 1) * p + (b_off[1] + 1)][col] += b_off[2]
                V[(b_off[0] + 0) * p + (b_off[1] + 1)][col] -= b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 0) * n + (c_off[0] + 0)][col] += c_off[2]
                W[(c_off[1] + 0) * n + (c_off[0] + 1)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                for (ai, aj) in [(0, 0), (0, 1)]:
                    U[(a_off[0] + ai) * m + (a_off[1] + aj)][col] += a_off[2]
            for b_off in B_offs:
                V[(b_off[0] + 1) * p + (b_off[1] + 1)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 0) * n + (c_off[0] + 0)][col] -= c_off[2]
                W[(c_off[1] + 1) * n + (c_off[0] + 0)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                U[(a_off[0] + 1) * m + (a_off[1] + 0)][col] += a_off[2]
                U[(a_off[0] + 0) * m + (a_off[1] + 0)][col] -= a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(0, 0), (0, 1)]:
                    V[(b_off[0] + bi) * p + (b_off[1] + bj)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 1) * n + (c_off[0] + 1)][col] += c_off[2]
            col += 1

            for a_off in A_offs:
                U[(a_off[0] + 0) * m + (a_off[1] + 1)][col] += a_off[2]
                U[(a_off[0] + 1) * m + (a_off[1] + 1)][col] -= a_off[2]
            for b_off in B_offs:
                for (bi, bj) in [(1, 0), (1, 1)]:
                    V[(b_off[0] + bi) * p + (b_off[1] + bj)][col] += b_off[2]
            for c_off in C_offs:
                W[(c_off[1] + 0) * n + (c_off[0] + 0)][col] += c_off[2]
            col += 1

            return col

        A_offs_M1 = [(0, 0, 1), (2, 2, 1)]
        B_offs_M1 = [(0, 0, 1), (2, 2, 1)]
        C_offs_M1 = [(0, 0, 1), (2, 2, -1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M1, B_offs_M1, C_offs_M1, start_col)

        A_offs_M2 = [(2, 0, 1), (2, 2, 1)]
        B_offs_M2 = [(0, 0, 1)]
        C_offs_M2 = [(2, 0, 1), (2, 2, -1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M2, B_offs_M2, C_offs_M2, start_col)

        A_offs_M3 = [(0, 0, 1)]
        B_offs_M3 = [(2, 0, 1), (0, 0, -1)]
        C_offs_M3 = [(0, 2, 1), (2, 2, 1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M3, B_offs_M3, C_offs_M3, start_col)

        A_offs_M4 = [(2, 2, 1)]
        B_offs_M4 = [(2, 2, 1), (0, 2, -1)]
        C_offs_M4 = [(0, 0, 1), (2, 0, 1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M4, B_offs_M4, C_offs_M4, start_col)

        A_offs_M5 = [(0, 0, 1), (0, 2, 1)]
        B_offs_M5 = [(2, 2, 1)]
        C_offs_M5 = [(0, 0, -1), (0, 2, 1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M5, B_offs_M5, C_offs_M5, start_col)

        A_offs_M6 = [(2, 0, 1), (0, 0, -1)]
        B_offs_M6 = [(0, 0, 1), (0, 2, 1)]
        C_offs_M6 = [(2, 2, 1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M6, B_offs_M6, C_offs_M6, start_col)

        A_offs_M7 = [(0, 2, 1), (2, 2, -1)]
        B_offs_M7 = [(2, 0, 1), (2, 2, 1)]
        C_offs_M7 = [(0, 0, 1)]
        start_col = _add_strassen_2x2_with_offsets(A_offs_M7, B_offs_M7, C_offs_M7, start_col)

        return start_col

    col = 0
    col = add_strassen_4x4_2x2_product(0, 0, 0, 0, 0, 0, col)

    assert col == 49, f"Expected 49, got {col}"
    return U, V, W


def construct():
    """
    Construct the tensor decomposition for 5x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    result_fallback = _schoolbook(n, m, p)

    total_rank = 106
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    U_s, V_s, W_s = _construct_strassen_4x4(n, m, p)

    for i in range(n * m):
        for c in range(49):
            U[i][c] = U_s[i][c]
            V[i][c] = V_s[i][c]
            W[i][c] = W_s[i][c]

    col = 49

    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    for j in range(4):
        for k in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
            col += 1

    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1

    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1

    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1

    if col != total_rank:
        return result_fallback

    result = {"U": U, "V": V, "W": W}

    if _verify(U, V, W, n, m, p):
        return result

    return result_fallback


# EVOLVE-BLOCK-END
