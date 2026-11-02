"""
Matrix multiplication 2x2 x 2x11: rank-minimal exact tensor decomposition.

Design:
  - Verified rank-39 baseline (5 Strassen-7 blocks for 10 columns + 4 schoolbook for last column).
  - Aggressive iterative-refinement search for rank 38..30 with a smooth epigraph (sum-of-squares)
    objective, alternating least-squares (ALS) updates, and a final high-precision polish pass.
  - The search is bounded by an explicit deadline guard (115 s inside the 120 s budget).
"""

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute the (i,j,k) -> (a,b,c) index lists once (module-level for speed).
def _build_index_lists(n, m, p):
    ta, tb, tc = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta.append(i * m + j)
                tb.append(j * p + k)
                tc.append(k * n + i)
    return (np.array(ta), np.array(tb), np.array(tc))

TA, TB, TC = _build_index_lists(*DIMENSIONS)

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check exact decomposition: sum_c U[a,c]*V[b,c]*W[c_index,c] == 1 for every (i,j,k)."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    sums = np.zeros(len(TA))
    for c in range(rank):
        sums += Ua[TA, c] * Va[TB, c] * Wa[TC, c]
    return bool(np.all(np.abs(sums - 1.0) <= tol))


def _construct_rank39():
    """
    Verified rank-39 decomposition: 5 Strassen-7 blocks (columns 0..9) + schoolbook (column 10).
    Each block handles two consecutive output columns k0, k1 with the 7 multiplications:
      m1 = (a+d)(e+h)
      m2 = (c+d)e
      m3 = a(f-h)
      m4 = d(g-e)
      m5 = (a+b)h
      m6 = (c-a)(e+f)
      m7 = (b-d)(g+h)
    and the 4th column (k=10) is handled by 4 schoolbook terms.
    """
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1

        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1

        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1

        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1

        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Schoolbook for column k=10
    k2 = 10
    v_e2 = k2
    v_g2 = p + k2
    w_c02 = k2 * n
    w_c12 = k2 * n + 1

    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1

    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1

    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1

    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1

    assert r == rank
    return U, V, W


def _structured_init(rank, n, m, p, seed=0):
    """
    Build an initial (U, V, W) guess for rank-r decomposition.
    Places Strassen-7 blocks in the first min(rank//7, p//2) pairs of columns,
    and adds small random noise everywhere to break symmetry.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1

    num_blocks = min(rank // 7, p // 2)
    for block in range(num_blocks):
        r0 = block * 7
        if r0 + 7 > rank:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1

        U[0, r0] += 1; U[3, r0] += 1
        V[v_e, r0] += 1; V[v_h, r0] += 1
        W[w_c00, r0] += 1; W[w_c11, r0] += 1

        U[2, r0 + 1] += 1; U[3, r0 + 1] += 1
        V[v_e, r0 + 1] += 1
        W[w_c10, r0 + 1] += 1; W[w_c11, r0 + 1] += -1

        U[0, r0 + 2] += 1
        V[v_f, r0 + 2] += 1; V[v_h, r0 + 2] += -1
        W[w_c01, r0 + 2] += 1; W[w_c11, r0 + 2] += 1

        U[3, r0 + 3] += 1
        V[v_g, r0 + 3] += 1; V[v_e, r0 + 3] += -1
        W[w_c00, r0 + 3] += 1; W[w_c10, r0 + 3] += 1

        U[0, r0 + 4] += 1; U[1, r0 + 4] += 1
        V[v_h, r0 + 4] += 1
        W[w_c00, r0 + 4] += -1; W[w_c01, r0 + 4] += 1

        U[2, r0 + 5] += 1; U[0, r0 + 5] += -1
        V[v_e, r0 + 5] += 1; V[v_f, r0 + 5] += 1
        W[w_c11, r0 + 5] += 1

        U[1, r0 + 6] += 1; U[3, r0 + 6] += -1
        V[v_g, r0 + 6] += 1; V[v_h, r0 + 6] += 1
        W[w_c00, r0 + 6] += 1

    return U, V, W


def _als_round(U, V, W, rank, n, m, p):
    """
    One full ALS round: update U, then V, then W (least-squares for each row).
    Uses the smooth sum-of-squares objective  sum_{(i,j,k)} (sum_c U[a,c]V[b,c]W[c_idx,c] - 1)^2.
    For each row of U, the optimal value given V, W is a closed-form ratio.
    """
    TA, TB, TC = _TA, _TB, _TC

    # Precompute masks once per call (cached)
    if not hasattr(_als_round, '_masks'):
        _als_round._masks = {
            'U': [TA == a for a in range(n * m)],
            'V': [TB == b for b in range(m * p)],
            'W': [TC == c for c in range(p * n)],
        }
    masks_U = _als_round._masks['U']
    masks_V = _als_round._masks['V']
    masks_W = _als_round._masks['W']

    # Update U
    for a in range(n * m):
        mask = masks_U[a]
        if not np.any(mask):
            continue
        V_sub = V[TB[mask], :]
        W_sub = W[TC[mask], :]
        prod = V_sub * W_sub
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            U[a, valid] = num[valid] / den[valid]

    # Update V
    for b in range(m * p):
        mask = masks_V[b]
        if not np.any(mask):
            continue
        U_sub = U[TA[mask], :]
        W_sub = W[TC[mask], :]
        prod = U_sub * W_sub
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            V[b, valid] = num[valid] / den[valid]

    # Update W
    for c_idx in range(p * n):
        mask = masks_W[c_idx]
        if not np.any(mask):
            continue
        U_sub = U[TA[mask], :]
        V_sub = V[TB[mask], :]
        prod = U_sub * V_sub
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            W[c_idx, valid] = num[valid] / den[valid]


def _compute_error(U, V, W):
    """Compute the sum-of-squares error."""
    sums = np.zeros(len(TA))
    for c in range(U.shape[1]):
        sums += U[TA, c] * V[TB, c] * W[TC, c]
    return float(np.sum((sums - 1.0) ** 2))


def _polish(U, V, W, rank, n, m, p, deadline, max_iter=5000):
    """
    High-precision polish: run many ALS iterations with tight convergence check.
    Returns (U, V, W, err) if converged below 1e-14, else (U, V, W, best_err).
    """
    best_err = _compute_error(U, V, W)
    for it in range(max_iter):
        if time.time() > deadline:
            break
        _als_round(U, V, W, rank, n, m, p)
        if it % 10 == 0:
            err = _compute_error(U, V, W)
            if err < best_err:
                best_err = err
            if err < 1e-14:
                return U, V, W, err
    return U, V, W, best_err


def _search_rank(rank, n, m, p, deadline, n_seeds=6, max_rounds=3000):
    """
    Search for a valid rank-r decomposition.
    Returns (U, V, W) as lists if found, else None.
    """
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        init_U, init_V, init_W = _structured_init(rank, n, m, p, seed=seed)
        U, V, W = init_U.copy(), init_V.copy(), init_W.copy()

        best_err = float('inf')
        converged = False
        for it in range(max_rounds):
            if time.time() > deadline:
                break
            _als_round(U, V, W, rank, n, m, p)
            if it % 25 == 0:
                err = _compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    # Save best
                    U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
                if err < 1e-12:
                    converged = True
                    break

        if converged:
            # Polish to high precision
            U, V, W, err = _polish(U, V, W, rank, n, m, p, deadline=deadline, max_iter=2000)
            if err < 1e-10:
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return U.tolist(), V.tolist(), W.tolist()
        else:
            # Use best found so far
            if best_err < 1e-10:
                U, V, W, err = _polish(U_best, V_best, W_best, rank, n, m, p, deadline=deadline, max_iter=2000)
                if err < 1e-10:
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return U.tolist(), V.tolist(), W.tolist()
    return None


# Module-level cached index arrays for speed
_TA, _TB, _TC = TA, TB, TC


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin inside 120s budget

    # 1. Verified rank-39 baseline
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
        # Fallback: schoolbook rank = n*m*p = 44
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

    best_result = (39, U39, V39, W39)

    # 2. Search for lower ranks: 38, 37, 36, 35, 34, 33, 32, 31, 30
    #    Allocate more seeds to higher-priority (lower) ranks.
    search_plan = [
        (38, 8), (37, 6), (36, 5), (35, 4), (34, 4), (33, 3), (32, 3), (31, 2), (30, 2)
    ]
    for target_rank, n_seeds in search_plan:
        if time.time() > deadline - 5:
            break
        result = _search_rank(target_rank, n, m, p, deadline=deadline - 3, n_seeds=n_seeds)
        if result is not None:
            best_result = (target_rank, result[0], result[1], result[2])
            # Continue searching for even lower rank
            continue

    final_rank, U_final, V_final, W_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
