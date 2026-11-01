"""
Matrix multiplication tensor decomposition for 2x5 * 5x6 -> 2x6
Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank below schoolbook.
"""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    # We'll partition the 5 inner dimensions into groups and the 6 outer into groups.
    # Strategy:
    # - j-groups: {0,1}, {2,3}, {4}
    # - k-groups: {0,1}, {2,3}, {4,5}
    # For each (j-group, k-group) pair, if j-group has size 2 and k-group has size 2,
    # use Strassen (rank 7). Otherwise use schoolbook.
    #
    # j-group {0,1} x k-group {0,1}: Strassen, rank 7
    # j-group {0,1} x k-group {2,3}: Strassen, rank 7
    # j-group {0,1} x k-group {4,5}: Strassen, rank 7
    # j-group {2,3} x k-group {0,1}: Strassen, rank 7
    # j-group {2,3} x k-group {2,3}: Strassen, rank 7
    # j-group {2,3} x k-group {4,5}: Strassen, rank 7
    # j-group {4} x k-group {0,1}: schoolbook 2x1x2, rank 4
    # j-group {4} x k-group {2,3}: schoolbook 2x1x2, rank 4
    # j-group {4} x k-group {4,5}: schoolbook 2x1x2, rank 4
    # Total rank = 6*7 + 3*4 = 42 + 12 = 54

    total_rank = 54

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(k, i):
        return k * n + i

    def ok():
        return time.time() - start < deadline

    def emit(u_list, v_list, w_list):
        nonlocal col
        if col >= total_rank:
            return
        if not ok():
            return
        for (idx, val) in u_list:
            U[idx][col] = val
        for (idx, val) in v_list:
            V[idx][col] = val
        for (idx, val) in w_list:
            W[idx][col] = val
        col += 1

    def strassen_2x2(j0, j1, k0, k1):
        """
        Decompose A[:, j0:j1+1] @ B[j0:j1+1, k0:k1+1] using Strassen.
        A sub-matrix rows: i=0,1; cols: j0, j1
        B sub-matrix rows: j0, j1; cols: k0, k1
        C sub-matrix rows: i=0,1; cols: k0, k1
        """
        a00, a01 = uidx(0, j0), uidx(0, j1)
        a10, a11 = uidx(1, j0), uidx(1, j1)
        b00, b01 = vidx(j0, k0), vidx(j0, k1)
        b10, b11 = vidx(j1, k0), vidx(j1, k1)
        c00, c01 = widx(k0, 0), widx(k1, 0)
        c10, c11 = widx(k0, 1), widx(k1, 1)

        # M1 = (A00 + A11)(B00 + B11)
        emit([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])

        # M2 = (A10 + A11) B00
        emit([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])

        # M3 = A00 (B01 - B11)
        emit([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])

        # M4 = A11 (B10 - B00)
        emit([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)])

        # M5 = (A00 + A01) B11
        emit([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])

        # M6 = (A10 - A00)(B00 + B01)
        emit([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])

        # M7 = (A01 - A11)(B10 + B11)
        emit([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    def schoolbook_2x1x2(j, k0, k1):
        """
        Decompose A[:, j] @ B[j, k0:k1+1] using schoolbook.
        A sub-matrix: 2x1 (rows i=0,1; col j)
        B sub-matrix: 1x2 (row j; cols k0, k1)
        C sub-matrix: 2x2 (rows i=0,1; cols k0, k1)
        """
        a0, a1 = uidx(0, j), uidx(1, j)
        b0, b1 = vidx(j, k0), vidx(j, k1)
        c00, c01 = widx(k0, 0), widx(k1, 0)
        c10, c11 = widx(k0, 1), widx(k1, 1)

        emit([(a0, 1)], [(b0, 1)], [(c00, 1)])
        emit([(a0, 1)], [(b1, 1)], [(c01, 1)])
        emit([(a1, 1)], [(b0, 1)], [(c10, 1)])
        emit([(a1, 1)], [(b1, 1)], [(c11, 1)])

    # Apply Strassen blocks for j-groups {0,1} and {2,3} with all k-groups
    for (j0, j1) in [(0, 1), (2, 3)]:
        for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
            if not ok():
                return {"U": U, "V": V, "W": W}
            strassen_2x2(j0, j1, k0, k1)

    # Apply schoolbook for j=4 with all k-groups
    for (k0, k1) in [(0, 1), (2, 3), (4, 5)]:
        if not ok():
            return {"U": U, "V": V, "W": W}
        schoolbook_2x1x2(4, k0, k1)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
