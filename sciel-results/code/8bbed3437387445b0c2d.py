# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Hard deadline guard: 630s wall-clock kill, leave 15s safety margin
_HARD_DEADLINE = time.time() + 615


def _compute_P_vectorized(support, weights):
    """Fully vectorized P computation for small n (n <= 56)."""
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0

    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    thresholds = 2.0 * support[None, :] - support[:, None]

    if n > 56:
        return _compute_P_fast(support, weights)

    S_4d = S[None, None, :, :]
    thr_4d = thresholds[:, :, None, None]
    mask = S_4d < thr_4d  # (k, l, i, j)
    W_4d = W[None, None, :, :]
    inner = np.einsum('klmn,klmn->kl', mask, W_4d)
    P = np.sum(w[:, None] * w[None, :] * inner)
    return float(P)


def _compute_P_fast(support, weights):
    """Faster computation using sorted pairwise sums (for larger n)."""
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cdf = np.cumsum(W_sorted)

    P = 0.0
    for k in range(n):
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            idx = np.searchsorted(S_sorted, thr, side='left')
            if idx > 0:
                P += w[k] * w[l] * W_cdf[idx - 1]
    return P


def _compute_P(support, weights):
    """Dispatch to appropriate P computation."""
    n = len(support)
    if n <= 56:
        return _compute_P_vectorized(support, weights)
    else:
        return _compute_P_fast(support, weights)


def _optimize_weights_nm(support, max_time=120.0, seed=42):
    """Optimize weights on fixed support using Nelder-Mead in log-space."""
    n = len(support)
    deadline = time.time() + max_time

    def neg_P_log(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P(support, w)

    best_P = -1.0
    best_w = np.ones(n) / n

    x0 = np.zeros(n)
    try:
        result = minimize(neg_P_log, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True, 'maxfev': 10000})
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = -result.fun
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    restarts = 20
    for i in range(restarts):
        if time.time() > deadline - 5:
            break
        x0 = rng.normal(0, 1.5, n)
        try:
            result = minimize(neg_P_log, x0, method='Nelder-Mead',
                              options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12,
                                       'adaptive': True, 'maxfev': 6000})
            w_opt = np.exp(result.x - result.x.max())
            w_opt = w_opt / w_opt.sum()
            P_opt = -result.fun
            if P_opt > best_P:
                best_P = P_opt
                best_w = w_opt
        except Exception:
            pass

    return best_w, best_P


def _optimize_weights_de(support, max_time=120.0, seed=42):
    """Optimize weights on fixed support using differential evolution (log-space)."""
    n = len(support)
    deadline = time.time() + max_time
    if n < 2:
        return np.ones(n) / max(n, 1), 0.0

    def neg_P_log(w_log):
        if time.time() > deadline:
            return 1e10
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1e10
        w = w / s
        return -_compute_P(support, w)

    bounds = [(-6.0, 6.0)] * n
    popsize = max(4, min(15, (2000 // max(n, 1)) + 4))
    try:
        result = differential_evolution(
            neg_P_log, bounds,
            seed=seed, maxiter=300, popsize=popsize, tol=1e-8,
            mutation=(0.3, 0.9), recombination=0.7, polish=True,
            init='sobol')
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = -result.fun
        return w_opt, P_opt
    except Exception:
        return np.ones(n) / n, -1.0


def _refine_support(support, max_n=48):
    """Insert midpoints between consecutive support points to refine geometry."""
    s = np.sort(np.asarray(support, dtype=np.float64))
    n = len(s)
    if n < 2:
        return s
    mids = (s[:-1] + s[1:]) / 2.0
    new = np.empty(2 * n - 1, dtype=np.float64)
    new[0::2] = s
    new[1::2] = mids
    if len(new) > max_n:
        # subsample evenly
        idx = np.linspace(0, len(new) - 1, max_n).astype(int)
        new = new[idx]
    return np.sort(new)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety: 600s

    best_support = np.arange(1, 25, dtype=np.float64)
    best_weights = np.ones(24, dtype=np.float64) / 24
    best_P = _compute_P(best_support, best_weights)

    # --- Strategy 1: geometric supports with optimized weights ---
    configs = [
        (16, 0.01, 2.0),
        (20, 0.01, 2.0),
        (24, 0.01, 2.0),
        (28, 0.01, 2.0),
        (32, 0.01, 2.0),
        (16, 0.01, 3.0),
        (20, 0.01, 3.0),
        (24, 0.01, 3.0),
        (28, 0.01, 3.0),
        (16, 0.1, 2.0),
        (20, 0.1, 2.0),
        (24, 0.1, 2.0),
        (20, 0.5, 2.0),
        (24, 0.5, 2.0),
        (16, 1.0, 2.0),
        (20, 1.0, 2.0),
        (24, 1.0, 2.0),
        (16, 0.001, 5.0),
        (20, 0.001, 5.0),
        (24, 0.001, 5.0),
        (16, 0.01, 4.0),
        (20, 0.01, 4.0),
        (24, 0.01, 4.0),
        (16, 0.5, 3.0),
        (20, 0.5, 3.0),
        (24, 0.5, 3.0),
        (16, 0.001, 10.0),
        (20, 0.001, 10.0),
        (16, 0.0001, 20.0),
        (20, 0.0001, 20.0),
        (16, 0.0, 2.0),
        (20, 0.0, 2.0),
        (24, 0.0, 2.0),
    ]

    for n, a, r in configs:
        if time.time() > deadline - 120:
            break
        if r <= 1.0:
            continue

        if a == 0.0:
            support = np.zeros(1)
            rest = r ** np.arange(1, n, dtype=np.float64)
            support = np.concatenate([support, rest])
        else:
            support = a * (r ** np.arange(n, dtype=np.float64))

        if np.any(support < 0) or not np.all(np.isfinite(support)):
            continue

        w_uniform = np.ones(n) / n
        P_uniform = _compute_P(support, w_uniform)
        if P_uniform < best_P * 0.3:
            continue

        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        max_opt_time = min(60.0, remaining * 0.5)
        w_opt, P_opt = _optimize_weights_nm(support, max_time=max_opt_time, seed=42)

        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()

    # --- Strategy 2: bimodal support family (small mass + geometric tail) ---
    # Structural idea: place w0 at a small value so X1+X2+X3 is often small,
    # while the tail lets 2*X4 be large.
    for n_tail, a, r, w0 in [
        (16, 0.01, 3.0, 0.3),
        (20, 0.01, 3.0, 0.3),
        (24, 0.01, 3.0, 0.3),
        (16, 0.05, 3.0, 0.4),
        (20, 0.05, 3.0, 0.4),
        (24, 0.05, 3.0, 0.4),
        (16, 0.1, 2.5, 0.3),
        (20, 0.1, 2.5, 0.3),
        (24, 0.1, 2.5, 0.3),
        (20, 0.01, 4.0, 0.2),
        (24, 0.01, 4.0, 0.2),
    ]:
        if time.time() > deadline - 90:
            break
        n = n_tail + 1
        tail = a * (r ** np.arange(n_tail, dtype=np.float64))
        support = np.concatenate([[0.0], tail])
        if np.any(support < 0) or not np.all(np.isfinite(support)):
            continue

        w_init = np.ones(n) / n
        w_init[0] = w0
        w_init = w_init / w_init.sum()
        P_init = _compute_P(support, w_init)
        if P_init < best_P * 0.4:
            continue

        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        max_opt_time = min(70.0, remaining * 0.5)
        w_opt, P_opt = _optimize_weights_nm(support, max_time=max_opt_time, seed=7)

        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()

    # --- Strategy 3: coarse-to-fine support refinement + differential evolution ---
    # Take the best support found so far, refine it (insert midpoints), and
    # re-optimize weights with DE (a different search family).
    if time.time() < deadline - 60:
        for refine_round in range(3):
            if time.time() > deadline - 40:
                break
            refined = _refine_support(best_support, max_n=48)
            if len(refined) < 2 or len(refined) == len(best_support):
                continue
            # initialize weights by linear interpolation of best_weights
            n_r = len(refined)
            if len(best_weights) == len(best_support):
                w_interp = np.interp(refined, best_support, best_weights)
                w_interp = np.maximum(w_interp, 1e-6)
                w_interp = w_interp / w_interp.sum()
            else:
                w_interp = np.ones(n_r) / n_r
            P_interp = _compute_P(refined, w_interp)

            remaining = deadline - time.time() - 10
            if remaining < 15:
                break
            max_opt_time = min(60.0, remaining * 0.6)
            w_opt, P_opt = _optimize_weights_de(refined, max_time=max_opt_time, seed=11 + refine_round)

            if P_opt > best_P:
                best_P = P_opt
                best_support = refined.copy()
                best_weights = w_opt.copy()

            # Also try NM on the refined support
            if time.time() < deadline - 25:
                w_opt2, P_opt2 = _optimize_weights_nm(refined, max_time=min(30.0, (deadline - time.time() - 20) * 0.5), seed=23)
                if P_opt2 > best_P:
                    best_P = P_opt2
                    best_support = refined.copy()
                    best_weights = w_opt2.copy()

    # --- Strategy 4: support scale search on the best geometric family ---
    # The problem is scale-free, but the discrete grid's relative spacing matters.
    # Try scaling the best support to shift the ratio structure.
    if time.time() < deadline - 30:
        base = best_support.copy()
        for scale in [0.2, 0.5, 2.0, 5.0]:
            if time.time() > deadline - 20:
                break
            cand = base * scale
            if np.any(cand < 0) or not np.all(np.isfinite(cand)):
                continue
            w = best_weights.copy()
            P = _compute_P(cand, w)
            if P > best_P:
                best_P = P
                best_support = cand.copy()
                best_weights = w.copy()

    # Final safety check
    if time.time() > deadline - 5:
        return best_support.astype(np.float64), best_weights.astype(np.float64)

    # Validate
    if len(best_support) != len(best_weights):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24
    if best_weights.sum() <= 1e-9:
        return np.arange(1, 25, dtype=np.float64), np.ones(24, dtype=np.float64) / 24

    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
