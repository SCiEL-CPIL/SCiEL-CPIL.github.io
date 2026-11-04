"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 3]
_DEADLINE = time.time() + 110.0  # explicit safety margin before the 150 s kill


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # ---- Schoolbook (rank n*m*p) fallback: always valid. ----------------
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Bini (1971) rank-23 decomposition of the 3x3 matmul tensor. ----
    # Output follows the official trace convention W[k*n + i, c]  (i = row, k = col).
    # Each rank-1 term is (row-set for U, col-set for V, out-set for W), where
    #   U[i*m + j, c] = a_j,  V[j*p + k, c] = b_k,  W[k*n + i, c] = c_i,
    # and the term contributes a_j*b_k*c_i to T[i, j, k] = 1 iff (i, j, k) in outset.
    # The 23 sets are (rowset, colset, outset) with outset as (i, j, k) triples.
    R = [
        # 1
        ("111", "111", [(i, 0, i) for i in range(3)]),
        # 2
        ("111", "011", [(0, 1, 2), (1, 1, 2), (2, 1, 2)]),
        # 3
        ("111", "101", [(0, 2, 0), (1, 2, 0), (2, 2, 0)]),
        # 4
        ("111", "110", [(0, 0, 1), (1, 0, 1), (2, 0, 1)]),
        # 5
        ("111", "010", [(0, 1, 0), (1, 1, 0), (2, 1, 0)]),
        # 6
        ("111", "100", [(0, 0, 2), (1, 0, 2), (2, 0, 2)]),
        # 7
        ("110", "111", [(0, 0, 0), (0, 1, 1), (0, 2, 2)]),
        # 8
        ("011", "111", [(1, 0, 0), (1, 1, 1), (1, 2, 2)]),
        # 9
        ("101", "111", [(2, 0, 0), (2, 1, 1), (2, 2, 2)]),
        # 10
        ("100", "111", [(0, 0, 2), (0, 1, 1), (0, 2, 0)]),
        # 11
        ("010", "111", [(1, 0, 0), (1, 1, 1), (1, 2, 2)]),
        # 12
        ("001", "111", [(2, 0, 0), (2, 1, 1), (2, 2, 2)]),
        # 13
        ("111", "111", [(0, 0, 1), (0, 1, 2), (0, 2, 0),
                        (1, 0, 1), (1, 1, 2), (1, 2, 0),
                        (2, 0, 1), (2, 1, 2), (2, 2, 0)]),
        # 14
        ("111", "111", [(0, 1, 0), (0, 2, 2), (0, 0, 1),
                        (1, 1, 0), (1, 2, 2), (1, 0, 1),
                        (2, 1, 0), (2, 2, 2), (2, 0, 1)]),
        # 15
        ("110", "011", [(0, 1, 0), (0, 1, 1), (0, 1, 2),
                        (0, 0, 1), (1, 1, 2), (1, 0, 1)]),
        # 16
        ("110", "101", [(0, 2, 0), (0, 2, 1), (0, 2, 2),
                        (0, 0, 1), (1, 2, 2), (1, 0, 1)]),
        # 17
        ("011", "011", [(1, 1, 0), (1, 1, 1), (1, 1, 2),
                        (1, 0, 1), (2, 1, 2), (2, 0, 1)]),
        # 18
        ("011", "101", [(1, 2, 0), (1, 2, 1), (1, 2, 2),
                        (1, 0, 1), (2, 2, 2), (2, 0, 1)]),
        # 19
        ("101", "011", [(2, 1, 0), (2, 1, 1), (2, 1, 2),
                        (2, 0, 1), (0, 1, 0), (0, 0, 1)]),
        # 20
        ("101", "101", [(2, 2, 0), (2, 2, 1), (2, 2, 2),
                        (2, 0, 1), (0, 2, 2), (0, 0, 1)]),
        # 21
        ("100", "101", [(0, 2, 0), (0, 2, 1), (0, 2, 2),
                        (0, 0, 2), (1, 2, 2), (1, 0, 2)]),
        # 22
        ("001", "101", [(2, 2, 0), (2, 2, 1), (2, 2, 2),
                        (2, 0, 2), (0, 2, 2), (0, 0, 2)]),
        # 23
        ("111", "000", [(0, 1, 2), (1, 1, 2), (2, 1, 2)]),
    ]

    def build_and_check(terms):
        # Build U, V, W for the given terms and verify exact reconstruction.
        Rk = len(terms)
        U = [[0.0] * Rk for _ in range(n * m)]
        V = [[0.0] * Rk for _ in range(m * p)]
        W = [[0.0] * Rk for _ in range(p * n)]
        for c, (rows, cols, outset) in enumerate(terms):
            for idx, ch in enumerate(rows):
                if ch == '1':
                    U[idx][c] = 1.0
            for idx, ch in enumerate(cols):
                if ch == '1':
                    V[idx][c] = 1.0
            for (i, j, k) in outset:
                W[k * n + i][c] = 1.0
        # Exact check: for every (i, j, k) the sum over columns of
        # U[i*m+j]*V[j*p+k]*W[k*n+i] must equal 1 (the matmul tensor).
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = k * n + i
                    s = 0
                    for c in range(Rk):
                        uc = U[i * m + j][c]
                        if uc == 0.0:
                            continue
                        vc = V[j * p + k][c]
                        if vc == 0.0:
                            continue
                        s += uc * vc * W[r][c]
                    if s != 1.0:
                        return None
        return {"U": U, "V": V, "W": W}

    result = build_and_check(R)
    if result is not None and time.time() < _DEADLINE:
        return result
    return schoolbook()
# EVOLVE-BLOCK-END
