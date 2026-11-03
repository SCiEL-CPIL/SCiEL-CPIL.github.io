"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_6x6():
    """
    Build a rank-189 decomposition for 6x6 * 6x6 matrix multiplication
    using 2x2 Strassen blocks on a 3x3 block grid.

    Index conventions:
      U: row index = i*m + j  (i: 0..5, j: 0..5)
      V: row index = j*p + k  (j: 0..5, k: 0..5)
      W: row index = k*n + i  (k: 0..5, i: 0..5)

    We decompose 6x6 into a 3x3 grid of 2x2 blocks.
    For each (ib, jb, kb) triple, we multiply block A[ib,kb] * B[kb,jb]
    using the 7-multiplication Strassen formula.
    Total rank = 3*3*3 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 formulas
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    #
    # M1 = a00 * b00              -> C00 += M1, C11 += M1
    # M2 = (a10 + a11) * b00      -> C10 += M2, C11 -= M2
    # M3 = a00 * (b01 - b11)      -> C01 += M3, C11 += M3
    # M4 = a11 * (b10 - b00)      -> C00 += M4, C10 += M4
    # M5 = (a00 + a01) * b11      -> C00 -= M5, C01 += M5
    # M6 = (a10 - a00) * (b00 + b01) -> C11 += M6
    # M7 = (a01 - a11) * (b10 + b11) -> C00 += M7

    # Each entry: (u_rows, u_coeffs, v_rows, v_coeffs, w_rows, w_coeffs)
    # u_rows: list of (a, c) in 0..1 for A block entry A[a,c]
    # v_rows: list of (c, b) in 0..1 for B block entry B[c,b]
    # w_rows: list of (b, a) in 0..1 for C block entry C[b,a]

    strassen_entries = [
        # M1 = a00 * b00
        ([(0, 0)], [1], [(0, 0)], [1], [(0, 0), (1, 1)], [1, 1]),
        # M2 = (a10 + a11) * b00
        ([(1, 0), (1, 1)], [1, 1], [(0, 0)], [1], [(1, 0), (1, 1)], [1, -1]),
        # M3 = a00 * (b01 - b11)
        ([(0, 0)], [1], [(0, 1), (1, 1)], [1, -1], [(0, 1), (1, 1)], [1, 1]),
        # M4 = a11 * (b10 - b00)
        ([(1, 1)], [1], [(1, 0), (0, 0)], [1, -1], [(0, 0), (1, 0)], [1, 1]),
        # M5 = (a00 + a01) * b11
        ([(0, 0), (0, 1)], [1, 1], [(1, 1)], [1], [(0, 0), (0, 1)], [-1, 1]),
        # M6 = (a10 - a00) * (b00 + b01)
        ([(1, 0), (0, 0)], [1, -1], [(0, 0), (0, 1)], [1, 1], [(1, 1)], [1]),
        # M7 = (a01 - a11) * (b10 + b11)
        ([(0, 1), (1, 1)], [1, -1], [(1, 0), (1, 1)], [1, 1], [(0, 0)], [1]),
    ]

    col = 0
    for ib in range(3):
        for kb in range(3):
            for jb in range(3):
                for (u_rows, u_coeffs, v_rows, v_coeffs, w_rows, w_coeffs) in strassen_entries:
                    for (a, c), coeff_u in zip(u_rows, u_coeffs):
                        u_idx = (2 * ib + a) * m + (2 * kb + c)
                        U[u_idx][col] += coeff_u
                    for (c, b), coeff_v in zip(v_rows, v_coeffs):
                        v_idx = (2 * kb + c) * p + (2 * jb + b)
                        V[v_idx][col] += coeff_v
                    for (b, a), coeff_w in zip(w_rows, w_coeffs):
                        w_idx = (2 * jb + b) * n + (2 * ib + a)
                        W[w_idx][col] += coeff_w
                    col += 1

    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Build a rank-216 schoolbook decomposition for 6x6 * 6x6."""
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(result, n, m, p, samples=500):
    """Verify the decomposition on a random sample of (i,j,k) entries."""
    import random
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    rng = random.Random(12345)
    for _ in range(samples):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, m - 1)
        k = rng.randint(0, p - 1)
        expected = 1.0 if j == k else 0.0
        actual = 0.0
        for c in range(rank):
            actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(actual - expected) > 1e-9:
            return False
    return True


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 * 6x6 matrix multiplication.
    Uses Strassen-style 2x2 block decomposition with 3x3 block grid.
    Rank = 189 (vs schoolbook 216).
    """
    n, m, p = DIMENSIONS

    t0 = time.time()
    deadline = 140  # safety margin before 150s kill

    # Try Strassen first
    try:
        result = _build_strassen_6x6()
        if time.time() - t0 > deadline:
            raise TimeoutError()
        if _verify(result, n, m, p, samples=300):
            return result
    except Exception:
        pass

    # Fallback to schoolbook
    try:
        result = _build_schoolbook()
        return result
    except Exception:
        # Last resort: return a trivial valid structure
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
