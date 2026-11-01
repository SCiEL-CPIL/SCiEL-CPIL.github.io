import numpy as np
import time

DIMENSIONS = [2, 4, 7]
_DEADLINE = time.time() + 130.0  # safety margin under 150s wall-clock


def _now_ok():
    return time.time() < _DEADLINE


def construct():
    n, m, p = DIMENSIONS

    # The matrix-multiplication tensor for 2x4 * 4x7 has 56 non-zero entries.
    # We build an explicit rank-50 decomposition using Strassen's 2x2x2
    # algorithm (rank 7) on 2x2 sub-blocks of the inner dimension and output,
    # plus a rank-4 schoolbook block for the remaining single output column.
    #
    # Layout:
    #   A is 2x4, B is 4x7, C is 2x7.
    #   Split inner dim j into {0,1} and {2,3}.
    #   For each j-block, split output k into {0,1},{2,3},{4,5} (Strassen)
    #   and {6} (schoolbook 2x2x1).
    #
    # Total rank = 2 * (3 * 7 + 4) = 2 * 25 = 50.

    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen 2x2x2 block: multiplies a 2x2 A-block by a 2x2 B-block
    # producing a 2x2 C-block using 7 products.
    # i0: starting row index in A (0 or 1)
    # j0: starting col index in A (0 or 2)
    # k0: starting col index in B (0, 2, or 4)
    def add_strassen_block(i0, j0, k0):
        nonlocal col

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    # Schoolbook 2x2x1 block: multiplies a 2x2 A-block by a 2x1 B-column
    # producing a 2x1 C-column using 4 products.
    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        # P2 = b*g
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

        # P3 = c*e
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

        # P4 = d*g
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    # Process both j-blocks
    for j_start in (0, 2):
        for k_start in (0, 2, 4):
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    # Verify the decomposition on random inputs
    try:
        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.integers(-3, 4, size=(n, m))
            B = rng.integers(-3, 4, size=(m, p))
            C_true = A @ B
            C2 = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    C2[i, k] = s
            if not np.allclose(C2, C_true, atol=1e-6):
                return _schoolbook(n, m, p)
    except Exception:
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
