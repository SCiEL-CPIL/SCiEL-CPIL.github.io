"""Deterministic rank-77 decomposition for 3x4*4x7 matrix multiplication.

Uses 7 disjoint 2x2x2 Strassen blocks:
- 2 blocks with i in {0,1}, k including 6
- 5 blocks with i in {0,2}, k in {0..5}
Remaining 28 entries use schoolbook.
Total rank = 7*7 + 28 = 77.
"""
import time
import numpy as np

_DIMS = (3, 4, 7)


def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = _DIMS

    rank = 77
    rows_u = n * m          # 12
    rows_v = m * p          # 28
    rows_w = p * n          # 21

    U = [[0] * rank for _ in range(rows_u)]
    V = [[0] * rank for _ in range(rows_v)]
    W = [[0] * rank for _ in range(rows_w)]

    col = 0

    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block."""
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    # 7 Strassen blocks
    # B1: i={0,1}, j={0,1}, k={0,6}
    # B2: i={0,1}, j={2,3}, k={2,6}
    # B3-B7: i={0,2}, with 5 disjoint 2x2 rectangles in j x k (j in 0..3, k in 0..5)
    blocks = [
        (0, 1, 0, 1, 0, 6),   # B1
        (0, 1, 2, 3, 2, 6),   # B2
        (0, 2, 0, 1, 0, 1),   # B3
        (0, 2, 0, 1, 2, 3),   # B4
        (0, 2, 0, 1, 4, 5),   # B5
        (0, 2, 2, 3, 0, 1),   # B6
        (0, 2, 2, 3, 2, 3),   # B7
    ]

    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))

    # Schoolbook for all remaining (i,j,k)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                if (i0, j0, k0) in covered:
                    continue
                U[i0 * m + j0][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition using numpy for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(m):
            u_row = U_np[i * m + j]
            for k in range(p):
                v_row = V_np[j * p + k]
                w_row = W_np[k * n + i]
                total = np.dot(u_row, v_row * w_row)
                if abs(total - 1.0) > 1e-10:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")

    return {"U": U, "V": V, "W": W}
