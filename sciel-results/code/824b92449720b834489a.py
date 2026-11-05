"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    def strassen_terms():
        # Standard Strassen 7 terms for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        T = []
        # M1
        T.append(([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1), ((1, 1), 1)]))
        # M2
        T.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)],
                 [((1, 0), 1), ((1, 1), -1)]))
        # M3
        T.append(([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)],
                 [((0, 1), 1), ((1, 1), 1)]))
        # M4
        T.append(([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)],
                 [((1, 0), 1), ((0, 0), 1)]))
        # M5
        T.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)],
                 [((0, 0), -1), ((0, 1), 1)]))
        # M6
        T.append(([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)],
                 [((1, 1), 1)]))
        # M7
        T.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1)]))
        return T

    ST = strassen_terms()

    # Build the rank-44 decomposition:
    # 4 Strassen blocks of rank 7 for the 2x2 sub-blocks of rows {0,1} x cols {0,1} and {2,3}
    # 4 schoolbook blocks of rank 4 for row 2 x cols {0,1} and {2,3}
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    def add_rank7(i0, j0, k0, col_start):
        for r in range(7):
            u_e, v_e, w_e = ST[r]
            c = col_start + r
            for (li, lj), val in u_e:
                U[(i0 + li) * m + (j0 + lj), c] += val
            for (lj, lk), val in v_e:
                V[(j0 + lj) * p + (k0 + lk), c] += val
            for (li, lk), val in w_e:
                W[(k0 + lk) * n + (i0 + li), c] += val
        return col_start + 7

    def add_rank4(i0, j0, k0, col_start):
        for t in range(4):
            c = col_start + t
            j = j0 + (t % 2)
            k = k0 + (t // 2)
            U[i0 * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i0, c] = 1
        return col_start + 4

    col = 0
    # Top 2 rows, cols 0-1
    col = add_rank7(0, 0, 0, col)
    # Top 2 rows, cols 2-3
    col = add_rank7(0, 0, 2, col)
    # Top 2 rows, cols 0-1 (for the second column pair of B)
    col = add_rank7(0, 2, 0, col)
    # Top 2 rows, cols 2-3
    col = add_rank7(0, 2, 2, col)
    # Bottom row (row 2), cols 0-1
    col = add_rank4(2, 0, 0, col)
    # Bottom row (row 2), cols 2-3
    col = add_rank4(2, 0, 2, col)
    # Bottom row (row 2), cols 0-1 (second j-pair)
    col = add_rank4(2, 2, 0, col)
    # Bottom row (row 2), cols 2-3
    col = add_rank4(2, 2, 2, col)

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    # Verify the decomposition
    def verify(U, V, W, rank):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_comp = np.zeros((n, p))
        for r in range(rank):
            ar = sum(U[i * m + j, r] * A[i, j] for i in range(n) for j in range(m))
            br = sum(V[j * p + k, r] * B[j, k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * ar * br
        max_err = float(np.max(np.abs(C_comp - C_expected)))
        return max_err

    max_err = verify(U, V, W, total_rank)
    if max_err > 1e-9:
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
