# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar


_DEADLINE = 630.0  # wall-clock kill; we return with a safety margin below this


def _compute_P_fast(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] using sorted pair-sums + searchsorted. O(n^2 log n)."""
    s = np.asarray(support, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return 0.0
    total = w.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return 0.0
    p = w / total

    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cumsum_probs = np.cumsum(sorted_probs)

    # thresholds[k,l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    flat_thresholds = thresholds.ravel()

    counts = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    F = np.zeros_like(flat_thresholds)
    mask = counts > 0
    F[mask] = cumsum_probs[counts[mask] - 1]
    F = F.reshape(n, n)

    weight_matrix = p[:, None] * p[None, :]
    P = np.sum(weight_matrix * F)
    return float(P)


def _two_point_p(p):
    """P for {0, a>0} with P(0)=p, P(a)=1-p. = (1-p)*p^2*(3-2p)."""
    q = 1.0 - p
    return q * p * p * (3.0 - 2.0 * p)


def _optimize_two_point():
    res = minimize_scalar(lambda p: -_two_point_p(p), bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    p_opt = float(res.x)
    return p_opt, _two_point_p(p_opt)


def _optimize_weights_fixed_support(support, initial_weights, deadline, max_iter=1500):
    """Refine weights (simplex) for a FIXED support: coordinate ascent + log-simplex NM."""
    s = np.asarray(support, dtype=np.float64).ravel()
    n = len(s)
    if n < 1:
        return np.asarray(initial_weights), 0.0
    w = np.asarray(initial_weights, dtype=np.float64).ravel().copy()
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 1e-9:
        w = np.full(n, 1.0 / n)
        tot = 1.0
    w = w / tot

    best_P = _compute_P_fast(s, w)
    best_w = w.copy()

    # --- coordinate ascent on the simplex ---
    step = 0.05
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for delta in (step, -step):
                    nw = w.copy()
                    nw[i] -= delta
                    nw[j] += delta
                    if nw[i] < 1e-9 or nw[j] < 1e-9:
                        continue
                    P = _compute_P_fast(s, nw)
                    if P > best_P:
                        best_P = P
                        best_w = nw.copy()
                        w = nw.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break

    # --- log-simplex Nelder-Mead refinement ---
    if time.time() < deadline - 20 and n >= 2:
        def neg_P_log(log_w):
            ew = np.exp(log_w - log_w.max())
            ew = ew / ew.sum()
            return -_compute_P_fast(s, ew)

        w_init = np.maximum(best_w, 1e-12)
        log_init = np.log(w_init)
        log_init = log_init - log_init.max()
        try:
            r = minimize(neg_P_log, log_init, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            ew = np.exp(r.x - r.x.max())
            ew = ew / ew.sum()
            P = _compute_P_fast(s, ew)
            if P > best_P:
                best_P = P
                best_w = ew.copy()
        except Exception:
            pass
    return best_w, float(best_P)


def _try_geometric_support(ratio, n_points, deadline):
    if ratio <= 1.0:
        return None, None, 0.0
    support = np.array([0.0] + [ratio**i for i in range(1, n_points)], dtype=np.float64)
    weights = np.array([0.5 * (0.5 ** i) for i in range(n_points)], dtype=np.float64)
    weights = weights / weights.sum()
    best_w, best_P = _optimize_weights_fixed_support(support, weights, deadline)
    return support, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before 630s kill

    best_P = -1.0
    best_support = None
    best_weights = None

    def _update(sup, wts, P):
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = np.asarray(sup, dtype=np.float64).ravel().copy()
            best_weights = np.asarray(wts, dtype=np.float64).ravel().copy()

    # ---- Phase 1: two-point analytic optimum ----
    p_opt, P_two = _optimize_two_point()
    _update(np.array([0.0, 1.0], dtype=np.float64),
            np.array([p_opt, 1.0 - p_opt], dtype=np.float64), P_two)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 2: geometric / powers-of-2 supports with optimized weights ----
    for ratio in [1.5, 2.0, 3.0, 4.0, 5.0, 10.0]:
        for n_points in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 40:
                break
            sup, wts, P = _try_geometric_support(ratio, n_points, deadline)
            if sup is not None and P > best_P:
                _update(sup, wts, P)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 3: add points to the best support (spare time buys extra search) ----
    if best_support is not None and time.time() < deadline - 30:
        sup = best_support.copy()
        wts = best_weights.copy()
        n = len(sup)
        max_val = float(sup.max())
        candidates = sorted(set([0.0, max_val * 0.5, max_val * 2.0, max_val * 5.0,
                                 max_val * 0.25, max_val * 10.0]))
        for new_pos in candidates:
            if time.time() > deadline - 30:
                break
            if np.any(np.abs(sup - new_pos) < 1e-9 * max(1.0, max_val)):
                continue
            new_sup = np.append(sup, new_pos)
            new_sup = np.sort(new_sup)
            # rebuild weights in sorted order
            order = np.argsort(np.append(sup, new_pos))
            new_wts = np.append(wts, 0.0)[order]
            for eps in [0.05, 0.1, 0.2, 0.3]:
                if time.time() > deadline - 30:
                    break
                # put eps on the NEW point (last after sort)
                cand_w = new_wts * (1.0 - eps)
                cand_w[-1] = eps
                cand_w = np.maximum(cand_w, 0.0)
                P = _compute_P_fast(new_sup, cand_w)
                if P > best_P:
                    opt_w, P2 = _optimize_weights_fixed_support(new_sup, cand_w, deadline)
                    if P2 > best_P:
                        _update(new_sup, opt_w, P2)
                    sup = best_support.copy()
                    wts = best_weights.copy()
                    n = len(sup)
                    max_val = float(sup.max())

    # ---- Phase 4: final weight polish on the best support ----
    if best_support is not None and time.time() < deadline - 10:
        opt_w, P2 = _optimize_weights_fixed_support(best_support, best_weights, deadline)
        if P2 > best_P:
            _update(best_support, opt_w, P2)

    # ---- validate / fallback ----
    if best_support is None or best_weights is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        best_support = np.asarray(best_support, dtype=np.float64).ravel()
        best_weights = np.asarray(best_weights, dtype=np.float64).ravel()
        mask = (best_support >= -1e-9) & (best_weights >= -1e-9) & \
               np.isfinite(best_support) & np.isfinite(best_weights)
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if len(best_support) == 0 or best_weights.sum() <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END
