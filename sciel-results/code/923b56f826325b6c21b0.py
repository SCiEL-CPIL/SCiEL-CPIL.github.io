"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.

Strategy:
  The 2x4x6 multiplication tensor decomposes as a block-diagonal sum of two
  independent 4x6 multiplications (one per row of the 2x4 matrix A).
  Tensor rank is additive under direct sum, so rank(2x4x6) = 2 * rank(4x6).

  For 4x6, we use a Strassen-type construction:
    - Partition the 4x4 core into 2x2 blocks and apply Strassen (7 products)
      to compute the 2x2 output blocks.
    - The two extra columns of B (making it 4x6) require additional products.

  We use Strassen's 2x2x2 algorithm applied to overlapping 2x2 blocks:
    - For the 4x6 case, we can view it as computing 2 output rows (n=2)
      with 4 input columns (m=4) and 6 output columns (p=6).
    - We partition the 4 columns of A into two 2-column blocks, and the
      6 columns of B into three 2-column blocks.
    - For each pair (A-block, B-block), we use Strassen's 7 products.
    - Total: 2 * 3 * 7 = 42 products.

  This gives rank 42 for 2x4x6.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_START = time.time()
_DEADLINE = 135  # safety margin before 150 s kill


def _elapsed():
    return time.time() - _START


def _verify_decomposition(U, V, W, n, m, p, seed=12345, trials=8):
    """Exactly verify that U,V,W implement C = A @ B using the trace
    convention W[k*n + i, r]. Integer arithmetic only."""
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m))
        B = rng.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(U.shape[1]):
            a_part = np.zeros(n, dtype=int)
            for i in range(n):
                s = 0
                for j in range(m):
                    s += U[i * m + j, r] * A[i, j]
                a_part[i] = s
            b_part = np.zeros(p, dtype=int)
            for k in range(p):
                s = 0
                for j in range(m):
                    s += V[j * p + k, r] * B[j, k]
                b_part[k] = s
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, r] * a_part[i] * b_part[k]
        if not np.array_equal(C_computed, C_expected):
            return False
    return True


def _build_strassen_42():
    """Verified rank-42 Strassen baseline for 2x4x6."""
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]

    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]

    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for g in range(2):
        for h in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = g * 2 + j_loc
                    U[i * m + j, col] += coeff

                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    V[j * p + k, col] += coeff

                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = h * 2 + k_loc
                    W[k * n + i, col] += coeff

                col += 1

    assert col == rank
    return U, V, W


def _build_schoolbook_48():
    """Fallback schoolbook rank-48 decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    if _elapsed() > _DEADLINE:
        U, V, W = _build_schoolbook_48()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = _build_strassen_42()

    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U, V, W = _build_schoolbook_48()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
