import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
_START = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _build_strassen_28():
    """Static rank-28 decomposition: 4 independent 2x2 blocks, each Strassen (rank 7)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms: (U coeffs, V coeffs, W coeffs) over local 2x2 indices
    # local idx: 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]

    for b in range(4):
        j_off = 2 * (b // 2)  # A column block offset: 0,2,0,2
        k_off = 2 * (b % 2)   # C column block offset: 0,0,2,2
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = strassen_terms[s]
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                a_idx = i * m + (j_off + j_local)
                U[a_idx][col] += coeff
            for y_idx, coeff in v_dict.items():
                i_local = y_idx // 2
                j_local = y_idx % 2
                b_idx = (j_off + i_local) * p + (k_off + j_local)
                V[b_idx][col] += coeff
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                c_idx = (k_off + k_local) * n + i
                W[c_idx][col] += coeff

    return U, V, W


def _verify(U, V, W, n=2, m=4, p=4, tol=1e-7):
    """Return (ok, max_err) for the decomposition U,V,W against the all-ones tensor."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Ue = Ua.reshape(n, m, -1)
    Ve = Va.reshape(m, p, -1)
    We = Wa.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ue, Ve, We)
    err = float(np.max(np.abs(T - 1.0)))
    return err < tol, err


def _residuals(x, n=2, m=4, p=4, rank=None):
    n, m, p = 2, 4, 4
    if rank is None:
        rank = x.size // ((n * m + m * p + p * n))
    U = x[:n * m * rank].reshape(n * m, rank)
    V = x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = x[n * m * rank + m * p * rank:].reshape(p * n, rank)
    Ue = U.reshape(n, m, rank)
    Ve = V.reshape(m, p, rank)
    We = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ue, Ve, We)
    return (T - 1.0).ravel()


def _try_rank(U28, V28, W28, target_rank, n=2, m=4, p=4, max_restarts=40, tol=1e-7):
    """Search for a decomposition of target_rank. Returns (U,V,W) if found else None."""
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total = size_U + size_V + size_W

    def x_from(U, V, W):
        return np.concatenate([np.asarray(U, float).ravel(),
                               np.asarray(V, float).ravel(),
                               np.asarray(W, float).ravel()])

    # Warm start from rank-28 (zero-pad to target_rank if smaller, else truncate)
    def warm_start():
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        k = min(target_rank, 28)
        U[:, :k] = np.asarray(U28, float)[:, :k]
        V[:, :k] = np.asarray(V28, float)[:, :k]
        W[:, :k] = np.asarray(W28, float)[:, :k]
        return x_from(U, V, W)

    x0_warm = warm_start()
    candidates = [x0_warm]
    # random restarts
    rng = np.random.default_rng(0)
    for _ in range(max_restarts):
        if _time_left() < 8.0:
            break
        candidates.append(rng.standard_normal(total) * 0.5)

    best_err = float('inf')
    for x0 in candidates:
        if _time_left() < 6.0:
            break
        try:
            res = opt.least_squares(_residuals, x0, args=(n, m, p, target_rank),
                                    method='lm', max_nfev=4000, xtol=1e-12, ftol=1e-12)
            err = float(np.linalg.norm(res.fun))
            if err < best_err:
                best_err = err
                best_x = res.x
            if best_err < tol:
                break
        except Exception:
            continue

    if best_err < tol:
        U = best_x[:size_U].reshape(n * m, target_rank)
        V = best_x[size_U:size_U + size_V].reshape(m * p, target_rank)
        W = best_x[size_U + size_V:].reshape(p * n, target_rank)
        # polish
        try:
            res = opt.least_squares(_residuals, best_x, args=(n, m, p, target_rank),
                                    method='lm', max_nfev=20000, xtol=1e-14, ftol=1e-14)
            if float(np.linalg.norm(res.fun)) < best_err:
                best_x = res.x
                best_err = float(np.linalg.norm(res.fun))
                U = best_x[:size_U].reshape(n * m, target_rank)
                V = best_x[size_U:size_U + size_V].reshape(m * p, target_rank)
                W = best_x[size_U + size_V:].reshape(p * n, target_rank)
        except Exception:
            pass
        if best_err < tol:
            return U.tolist(), V.tolist(), W.tolist()
    return None


def construct():
    n, m, p = DIMENSIONS
    # Guaranteed rank-28 fallback
    U28, V28, W28 = _build_strassen_28()
    ok, err = _verify(U28, V28, W28)
    best = (28, U28, V28, W28)
    if not ok:
        # should not happen; still return it
        return {"U": U28, "V": V28, "W": W28}

    # Try to reduce rank below 28
    for target in (27, 26, 25, 24, 23, 22, 21, 20):
        if _time_left() < 15.0:
            break
        res = _try_rank(U28, V28, W28, target, n, m, p)
        if res is not None:
            U, V, W = res
            ok2, err2 = _verify(U, V, W)
            if ok2:
                best = (target, U, V, W)
                break  # keep searching lower in next iterations? we want minimal, so keep going
                # but we break to keep going lower; actually continue to try lower ranks
                # We'll continue the loop to attempt even lower ranks.

    # Continue attempting lower ranks after a success to find minimal
    # (the loop above breaks on first success; re-run descending to find minimal)
    # Redo: iterate all and keep the minimal found.
    found = {}
    for target in range(27, 19, -1):
        if _time_left() < 12.0:
            break
        if target in found:
            continue
        res = _try_rank(U28, V28, W28, target, n, m, p)
        if res is not None:
            U, V, W = res
            if _verify(U, V, W)[0]:
                found[target] = (U, V, W)

    if found:
        best_rank = min(found.keys())
        U, V, W = found[best_rank]
        return {"U": U, "V": V, "W": W}

    return {"U": best[1], "V": best[2], "W": best[3]}
