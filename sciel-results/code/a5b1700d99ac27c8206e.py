"""
Matrix multiplication 2x2 x 2x11: rank-minimal exact tensor decomposition.

Design (combined):
  - Verified rank-39 baseline (5 Strassen-7 blocks for 10 columns + 4 schoolbook for last column).
  - Iterative-refinement search for rank 38..30 using a smooth sum-of-squares objective,
    alternating least-squares (ALS) updates, and a high-precision polish pass.
  - NEW: structured initializations that seed SHARED (cross-column) terms, so the search can
    escape the block-diagonal (per-pair Strassen) structure that caps rank at 39. A rank-38
    solution must share a term across two column pairs, which the per-block structure cannot
    represent; seeding such a term lets ALS discover non-block-decomposable low-rank solutions.
  - Bounded by an explicit deadline guard (115 s inside the 120 s budget).
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
_TA, _TB, _TC = TA, TB, TC


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

        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1

        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1

        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1

        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1

        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1

        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1

        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

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


def _strassen_block(U, V, W, r0, k0, k1, p):
    """Write the 7 Strassen terms for column pair (k0, k1) starting at rank column r0."""
    v_e = k0
    v_f = k1
    v_g = p + k0
    v_h = p + k1
    w_c00 = k0 * 2
    w_c10 = k0 * 2 + 1
    w_c01 = k1 * 2
    w_c11 = k1 * 2 + 1

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


def _structured_init(rank, n, m, p, seed=0, mode='block'):
    """
    Build an initial (U, V, W) guess for rank-r decomposition.
    mode='block'  : per-pair Strassen blocks + small noise (original).
    mode='cross'  : rank-39 baseline with one cross-column (shared) term seeded, to break
                    the block-diagonal structure; small noise everywhere.
    mode='random' : pure random.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1

    if mode == 'random':
        return U, V, W

    if mode == 'cross':
        # Start from the verified rank-39 baseline (block-diagonal).
        U39, V39, W39 = _construct_rank39()
        U39 = np.array(U39, dtype=float)
        V39 = np.array(V39, dtype=float)
        W39 = np.array(W39, dtype=float)
        if rank >= 39:
            U[:, :39] += U39
            V[:, :39] += V39
            W[:, :39] += W39
            # Seed a cross-column term spanning two column pairs (indices 39..rank-1).
            # A genuine rank-38 solution must share a term across pairs; this gives ALS
            # a foothold in the non-block-decomposable region.
            extra = rank - 39
            for t in range(extra):
                c = 39 + t
                # spread across several (a,b,c_idx) entries to encourage cross-column coupling
                a_idx = [0, 3, (t * 2) % 4]
                for a in a_idx:
                    U[a, c] += 0.5
                for b in [0, p, 1, p + 1]:
                    V[b, c] += 0.5
                for w in [0, 1, 2, 3, 20, 21]:
                    W[w, c] += 0.5
            # small noise to break exact symmetry on the cross term
            if extra > 0:
                U[:, 39:] += rng.randn(n * m, extra) * 0.05
                V[:, 39:] += rng.randn(m * p, extra) * 0.05
                W[:, 39:] += rng.randn(p * n, extra) * 0.05
            return U, V, W
        # rank < 39: fall through to block packing below
        mode = 'block'

    # mode == 'block'
    num_blocks = min(rank // 7, p // 2)
    for block in range(num_blocks):
        r0 = block * 7
        if r0 + 7 > rank:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        _strassen_block(U, V, W, r0, k0, k1, p)
    return U, V, W


def _als_round(U, V, W, rank, n, m, p):
    """One full ALS round (U, V, W) for the smooth sum-of-squares objective."""
    TA, TB, TC = _TA, _TB, _TC
    # Precompute: for each index a, the list of tensor entries where TA==a
    # We use the fact that each (a,b,c) triple appears exactly once
    # For U[a]: sum over all (j,k) such that a=i*m+j -> entries where TA==a
    # Precompute index lists once
    if not hasattr(_als_round, '_idx'):
        idx_U = {}
        idx_V = {}
        idx_W = {}
        for t in range(len(TA)):
            a, b, c = TA[t], TB[t], TC[t]
            idx_U.setdefault(a, []).append(t)
            idx_V.setdefault(b, []).append(t)
            idx_W.setdefault(c, []).append(t)
        _als_round._idx = (idx_U, idx_V, idx_W)
    idx_U, idx_V, idx_W = _als_round._idx

    # Update U
    for a in range(n * m):
        if a not in idx_U:
            continue
        ts = idx_U[a]
        prod = V[TB[ts], :] * W[TC[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            U[a, valid] = num[valid] / den[valid]

    # Update V
    for b in range(m * p):
        if b not in idx_V:
            continue
        ts = idx_V[b]
        prod = U[TA[ts], :] * W[TC[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            V[b, valid] = num[valid] / den[valid]

    # Update W
    for c_idx in range(p * n):
        if c_idx not in idx_W:
            continue
        ts = idx_W[c_idx]
        prod = U[TA[ts], :] * V[TB[ts], :]
        num = prod.sum(axis=0)
        den = (prod ** 2).sum(axis=0)
        valid = den > 1e-14
        if np.any(valid):
            W[c_idx, valid] = num[valid] / den[valid]


def _compute_error(U, V, W):
    """Sum-of-squares error of the decomposition."""
    sums = np.zeros(len(TA))
    for c in range(U.shape[1]):
        sums += U[TA, c] * V[TB, c] * W[TC, c]
    return float(np.sum((sums - 1.0) ** 2))


def _polish(U, V, W, rank, n, m, p, deadline, max_iter=5000):
    """High-precision polish: many ALS iterations with tight convergence check."""
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


def _refine_once(U, V, W, rank, n, m, p, deadline, max_rounds=3000):
    """
    Run ALS from (U,V,W); return a verified (U,V,W) list if the objective reaches ~0, else None.
    """
    best_err = float('inf')
    U_best = V_best = W_best = None
    converged = False
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        _als_round(U, V, W, rank, n, m, p)
        if it % 25 == 0:
            err = _compute_error(U, V, W)
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            if err < 1e-12:
                converged = True
                break

    candidates = []
    if converged:
        candidates.append((U, V, W))
    if best_err < 1e-10 and U_best is not None:
        candidates.append((U_best, V_best, W_best))

    for Uc, Vc, Wc in candidates:
        if time.time() > deadline:
            break
        Uf, Vf, Wf, err = _polish(Uc, Vc, Wf if False else Wc, rank, n, m, p,
                                  deadline=deadline, max_iter=2000)
        if err < 1e-10:
            if _verify(Uf.tolist(), Vf.tolist(), Wf.tolist(), n, m, p):
                return Uf.tolist(), Vf.tolist(), Wf.tolist()
    return None


def _svd_reduce(U, V, W, rank, n, m, p, tol=1e-10):
    """
    Try to reduce rank by detecting linearly dependent columns via SVD.
    Returns (U_r, V_r, W_r) with reduced rank if possible, else original.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape[1] <= 1:
        return U, V, W

    # Check if any columns are (nearly) linearly dependent
    # Build the tensor residual structure: for each rank column c, the contribution
    # is U[:,c] ⊗ V[:,c] ⊗ W[:,c]. Two columns c1, c2 are "redundant" if their
    # outer products are linearly dependent in a specific sense.
    # Simpler approach: check if the combined tensor has rank < rank.
    # The tensor T[a,b,c_idx] = sum_r U[a,r]*V[b,r]*W[c_idx,r]
    # We can check rank by trying to remove columns one at a time.
    
    # Try greedy removal: for each column, check if removing it still gives exact solution
    # This is expensive, so use SVD of the "factor matrix" approach.
    # Alternative: check if columns of U (or V or W) are linearly dependent
    # AND the corresponding "other" factors compensate.
    
    # Practical approach: try each column removal and verify
    current_rank = Ua.shape[1]
    for col_to_remove in range(current_rank):
        if time.time() > _svd_reduce._deadline if hasattr(_svd_reduce, '_deadline') else 999999:
            break
        # Build reduced matrices
        mask = np.ones(current_rank, dtype=bool)
        mask[col_to_remove] = False
        Ur = Ua[:, mask]
        Vr = Va[:, mask]
        Wr = Wa[:, mask]
        new_rank = mask.sum()
        if new_rank < 2:
            continue
        # Quick check: compute error
        sums = np.zeros(len(TA))
        for c in range(new_rank):
            sums += Ur[TA, c] * Vr[TB, c] * Wr[TC, c]
        err = np.max(np.abs(sums - 1.0))
        if err < tol:
            # Verify fully
            if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p):
                return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return U, V, W


def _search_rank(rank, n, m, p, deadline, n_seeds=6, max_rounds=3000):
    """
    Search for a valid rank-r decomposition across several structured init modes.
    Returns (U, V, W) as lists if found, else None.
    """
    # Cross-column modes are the new hypothesis for escaping rank 39; give them priority.
    mode_cycle = ['cross', 'block', 'random', 'cross']
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        mode = mode_cycle[seed % len(mode_cycle)]
        init_U, init_V, init_W = _structured_init(rank, n, m, p, seed=seed, mode=mode)
        U, V, W = init_U.copy(), init_V.copy(), init_W.copy()
        result = _refine_once(U, V, W, rank, n, m, p, deadline=deadline, max_rounds=max_rounds)
        if result is not None:
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin inside the 120 s budget

    # 1. Verified rank-39 baseline
    U39, V39, W39 = _construct_rank39()
    if not _verify(U39, V39, W39, n, m, p):
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

    # 2. Try SVD-based rank reduction on the rank-39 baseline
    if time.time() < deadline - 10:
        _svd_reduce._deadline = time.time() + 10
        U_r, V_r, W_r = _svd_reduce(U39, V39, W39, 39, n, m, p, tol=1e-10)
        if len(U_r[0]) < 39:
            best_result = (len(U_r[0]), U_r, V_r, W_r)

    # 3. Search for lower ranks. Concentrate effort on 38/37 (most likely improvements),
    #    using cross-column structured inits to escape the block-diagonal structure.
    search_plan = [
        (38, 10), (37, 10), (36, 6), (35, 4), (34, 3), (33, 2), (32, 2), (31, 1), (30, 1)
    ]
    for target_rank, n_seeds in search_plan:
        if time.time() > deadline - 5:
            break
        if target_rank >= best_result[0]:
            continue
        result = _search_rank(target_rank, n, m, p, deadline=deadline - 3,
                              n_seeds=n_seeds, max_rounds=3000)
        if result is not None:
            best_result = (target_rank, result[0], result[1], result[2])
            # Continue searching for even lower rank.

    final_rank, U_final, V_final, W_final = best_result
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
