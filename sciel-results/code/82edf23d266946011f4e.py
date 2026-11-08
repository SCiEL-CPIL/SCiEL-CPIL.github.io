import time

_DIMENSIONS = (3, 4, 7)
_DEADLINE = 135.0


def _build_factors():
    n, m, p = _DIMENSIONS
    total_entries = n * m * p  # 84
    rank = total_entries - 6 * 1  # 84 - 6 = 78

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # The 6 Strassen 2x2x2 blocks we use.
    # (i0, i1, j0, j1, k0, k1)
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    covered = set()
    col = 0

    # Helper to place a single rank-1 term at column c.
    def place(u_row, v_row, w_row, c, u_val=1, v_val=1, w_val=1):
        U[u_row][c] = u_val
        V[v_row][c] = v_val
        W[w_row][c] = w_val

    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        #     -> C[i0,k0] +, C[i1,k1] +
        place(i0 * m + j0, j0 * p + k0, k0 * n + i0, col)
        place(i1 * m + j1, j0 * p + k0, k0 * n + i0, col)
        place(i0 * m + j0, j1 * p + k1, k0 * n + i0, col)
        place(i1 * m + j1, j1 * p + k1, k0 * n + i0, col)
        place(i0 * m + j0, j0 * p + k0, k1 * n + i1, col)
        place(i1 * m + j1, j0 * p + k0, k1 * n + i1, col)
        place(i0 * m + j0, j1 * p + k1, k1 * n + i1, col)
        place(i1 * m + j1, j1 * p + k1, k1 * n + i1, col)
        col += 1

        # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        #     -> C[i1,k0] +, C[i1,k1] -
        place(i1 * m + j0, j0 * p + k0, k0 * n + i1, col)
        place(i1 * m + j1, j0 * p + k0, k0 * n + i1, col)
        place(i1 * m + j0, j0 * p + k0, k1 * n + i1, col, w_val=-1)
        place(i1 * m + j1, j0 * p + k0, k1 * n + i1, col, w_val=-1)
        col += 1

        # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        #     -> C[i0,k1] +, C[i1,k1] +
        place(i0 * m + j0, j0 * p + k1, k1 * n + i0, col)
        place(i0 * m + j0, j1 * p + k1, k1 * n + i0, col, v_val=-1)
        place(i0 * m + j0, j0 * p + k1, k1 * n + i1, col)
        place(i0 * m + j0, j1 * p + k1, k1 * n + i1, col, v_val=-1)
        col += 1

        # M4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        #     -> C[i0,k0] +, C[i1,k0] +
        place(i1 * m + j1, j1 * p + k0, k0 * n + i0, col)
        place(i1 * m + j1, j0 * p + k0, k0 * n + i0, col, v_val=-1)
        place(i1 * m + j1, j1 * p + k0, k0 * n + i1, col)
        place(i1 * m + j1, j0 * p + k0, k0 * n + i1, col, v_val=-1)
        col += 1

        # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        #     -> C[i0,k0] -, C[i0,k1] +
        place(i0 * m + j0, j1 * p + k1, k0 * n + i0, col, w_val=-1)
        place(i0 * m + j1, j1 * p + k1, k0 * n + i0, col, w_val=-1)
        place(i0 * m + j0, j1 * p + k1, k1 * n + i0, col)
        place(i0 * m + j1, j1 * p + k1, k1 * n + i0, col)
        col += 1

        # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        #     -> C[i1,k1] +
        place(i1 * m + j0, j0 * p + k0, k1 * n + i1, col)
        place(i0 * m + j0, j0 * p + k0, k1 * n + i1, col, u_val=-1)
        place(i1 * m + j0, j0 * p + k1, k1 * n + i1, col)
        place(i0 * m + j0, j0 * p + k1, k1 * n + i1, col, u_val=-1)
        col += 1

        # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        #     -> C[i0,k0] +
        place(i0 * m + j1, j1 * p + k0, k0 * n + i0, col)
        place(i1 * m + j1, j1 * p + k0, k0 * n + i0, col, u_val=-1)
        place(i0 * m + j1, j1 * p + k1, k0 * n + i0, col)
        place(i1 * m + j1, j1 * p + k1, k0 * n + i0, col, u_val=-1)
        col += 1

        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook terms for all entries not covered by the Strassen blocks.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    if col != rank:
        raise AssertionError(f"expected rank {rank}, got {col}")
    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                total = 0
                for c in range(rank):
                    total += u_row[c] * v_row[c] * w_row[c]
                if total != 1:
                    raise ValueError(
                        f"decomposition failed at ({i},{j},{k}): got {total}, expected 1"
                    )


def construct():
    t0 = time.time()
    n, m, p = _DIMENSIONS
    U, V, W = _build_factors()
    rank = len(U[0])
    _verify(U, V, W, n, m, p, rank)
    if time.time() - t0 > _DEADLINE:
        pass
    return {"U": U, "V": V, "W": W}
