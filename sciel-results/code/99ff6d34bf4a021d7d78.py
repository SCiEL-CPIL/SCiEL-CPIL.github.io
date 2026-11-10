# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Hard deadline guard: 630s wall-clock kill, leave 15s safety margin
_HARD_DEADLINE = time.time() + 615


def _compute_P_fast(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure (vectorized)."""
    n = len(support)
    w = weights / weights.sum()
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0  # s+s+s < 2*s*s is s < s for s>0, false; for s=0 it's 0<0, false

    S = support[:, None] + support[None, :]  # n x n
    W = np.outer(w, w)  # n x n

    # thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]  # shape (n, n), index [k, l]

    # For each threshold, compute sum of W[i,j] where S[i,j] < threshold
    # Vectorize over k: for each k, we have n thresholds
    P = 0.0
    for k in range(n):
        thrs = thresholds[k, :]  # shape (n,)
        # For each l, count W where S < thr[l]
        # Use searchsorted on sorted S_flat
        # But doing it per k is O(n * n log(n^2)) which is OK for n <= 64
        pass

    # Alternative fully vectorized approach for small n:
    # P = sum_k sum_l w_k w_l * (W * (S < 2*s_l - s_k)).sum()
    # = sum_k w_k * sum_l w_l * F(2*s_l - s_k)
    # where F(t) = sum_{i,j: s_i+s_j < t} w_i w_j

    # Sort S and build CDF
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cdf = np.cumsum(W_sorted)

    P = 0.0
    for k in range(n):
        wk = w[k]
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            idx = np.searchsorted(S_sorted, thr, side='left')
            if idx > 0:
                P += wk * w[l] * W_cdf[idx - 1]
    return P


def _compute_P_vectorized(support, weights):
    """Fully vectorized P computation for small n (n <= 48)."""
    n = len(support)
    w = weights / weights.sum()
    if n <= 1:
        return 0.0

    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    # thresholds[k, l] = 2*s_l - s_k, shape (n, n)
    thresholds = 2.0 * support[None, :] - support[:, None]

    # For each (k,l) pair, we need sum of W[i,j] where S[i,j] < thresholds[k,l]
    # This is a 4D operation. For n <= 48, n^4 = ~5M, feasible.
    # Vectorize: create mask of shape (n, n, n, n) where mask[k, l, i, j] = S[i,j] < 2*s_l - s_k
    # This uses n^4 memory. For n=48: 48^4 = 5.3M floats = 42 MB. OK.
    # For n=64: 64^4 = 16.8M floats = 134 MB. Risky.
    if n > 56:
        return _compute_P_fast(support, weights)

    # Build 4D mask
    # S[i,j] expanded: S[None, None, :, :]
    # thresholds[k,l] expanded: thresholds[:, :, None, None]
    # mask shape: (n, n, n, n)
    S_4d = S[None, None, :, :]
    thr_4d = thresholds[:, :, None, None]
    mask = S_4d < thr_4d  # (k, l, i, j)

    # W[i,j] expanded: W[None, None, :, :]
    W_4d = W[None, None, :, :]
    # Weighted sum over (i,j) for each (k,l):
    # inner = (W_4d * mask).sum(axis=(2, 3))  # shape (n, n)
    inner = np.einsum('klmn,klmn->kl', mask, W_4d)

    # P = sum_k sum_l w_k w_l * inner[k,l]
    P = np.sum(w[:, None] * w[None, :] * inner)
    return float(P)


def _compute_P(support, weights):
    """Dispatch to appropriate P computation."""
    n = len(support)
    if n <= 56:
        return _compute_P_vectorized(support, weights)
    else:
        return _compute_P_fast(support, weights)


def _optimize_weights_fixed_support(support, max_time=120.0, seed=42):
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

    # Initial: uniform
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

    # Random restarts
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


def _optimize_support_and_weights(n, initial_support, max_time=180.0, seed=42):
    """Jointly optimize support and weights."""
    deadline = time.time() + max_time

    # Parameterize: support as exp(log_support), weights as exp(log_weights)
    # But support must be >= 0, so use softplus or exp
    def neg_P_params(params):
        if time.time() > deadline:
            return 1e10
        n_local = len(initial_support)
        log_s = params[:n_local]
        log_w = params[n_local:]
        s = np.exp(log_s - log_s.max())  # ensures positive
        w = np.exp(log_w - log_w.max())
        s = s / s.sum()  # normalize support to [0,1] range
        w = w / w.sum()
        # Actually we want support in a meaningful range. Let's use:
        # s_i = exp(log_s_i) directly, but constrain to [0, inf)
        # Better: s_i = softplus(log_s_i) = log(1 + exp(log_s_i))
        s = np.log1p(np.exp(log_s))
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        return -_compute_P(s, w)

    x0_s = np.log(np.maximum(initial_support, 1e-10) + 1.0)  # inverse softplus
    x0_w = np.zeros(n)
    x0 = np.concatenate([x0_s, x0_w])

    best_P = -1.0
    best_params = x0.copy()

    try:
        result = minimize(neg_P_params, x0, method='Nelder-Mead',
                          options={'maxiter': 12000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True, 'maxfev': 15000})
        P_opt = -result.fun
        if P_opt > best_P:
            best_P = P_opt
            best_params = result.x
    except Exception:
        pass

    # Extract
    log_s = best_params[:n]
    log_w = best_params[n:]
    s = np.log1p(np.exp(log_s))
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()

    return s, w, best_P


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

    # Strategy 1: Geometric supports with optimized weights
    configs = [
        # (n, a, r)
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
        # Heavier tails
        (16, 0.001, 10.0),
        (20, 0.001, 10.0),
        (16, 0.0001, 20.0),
        (20, 0.0001, 20.0),
        # Include zero
        (16, 0.0, 2.0),
        (20, 0.0, 2.0),
        (24, 0.0, 2.0),
    ]

    for n, a, r in configs:
        if time.time() > deadline - 90:
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

        # Quick uniform check
        w_uniform = np.ones(n) / n
        P_uniform = _compute_P(support, w_uniform)
        if P_uniform < best_P * 0.3:
            continue

        # Optimize weights
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        max_opt_time = min(60.0, remaining * 0.6)
        w_opt, P_opt = _optimize_weights_fixed_support(support, max_time=max_opt_time, seed=42)

        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()

    # Strategy 2: Joint optimization of support and weights for a few sizes
    for n in [12, 16, 20]:
        if time.time() > deadline - 60:
            break
        # Initial support: geometric
        initial_support = 0.01 * (3.0 ** np.arange(n, dtype=np.float64))
        remaining = deadline - time.time() - 10
        if remaining < 15:
            break
        s_opt, w_opt, P_opt = _optimize_support_and_weights(
            n, initial_support, max_time=min(45.0, remaining * 0.7), seed=123)
        if P_opt > best_P:
            best_P = P_opt
            best_support = s_opt.copy()
            best_weights = w_opt.copy()

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

    # Ensure support is non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
