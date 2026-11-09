# EVOLVE-BLOCK-START
import time
import numpy as np


def _vertex_scores(P):
    """Per-vertex equidistance score in [0,1] (1 = perfect cluster). Higher is better."""
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    dist = np.sqrt((D * D).sum(-1))
    np.fill_diagonal(dist, np.inf)
    s = np.sort(dist, axis=1)  # (n, n-1), d_0 .. d_{n-2}
    gaps = s[:, 1:] - s[:, :-1]
    k = np.argmin(gaps, axis=1)
    Dv = (s[np.arange(n), k] + s[np.arange(n), k + 1]) / 2.0
    any_small = (s < 0.05).any(axis=1)
    D_small = Dv < 0.1
    idx = np.arange(n)
    i0 = np.where(k == 0, 0, np.where(k == n - 2, n - 5, k - 1))
    sel = s[:, i0:i0 + 4]
    r = np.maximum(sel / Dv[:, None], Dv[:, None] / sel)
    raw = r.mean(axis=1)
    score = np.minimum(1.0, 1.0 / np.maximum(raw, 1e-12))
    score[any_small | D_small] = 0.0
    return score


def _convexity_penalty(P):
    """0 if strictly convex (consistent nonzero turns, total ~ +-2pi, no repeats)."""
    n = P.shape[0]
    e = np.roll(P, -1, axis=0) - P
    cross = e[:, 0] * np.roll(e, -1, axis=0)[:, 1] - e[:, 1] * np.roll(e, -1, axis=0)[:, 0]
    if np.any(cross == 0):
        return 1e6
    if np.any(cross > 0) and np.any(cross < 0):
        return 1e6
    ang = np.arctan2(cross, (e * np.roll(e, -1, axis=0)).sum(-1))
    total = np.sum(ang)
    if abs(abs(total) - 2 * np.pi) > 1e-2:
        return 1e6
    minsep = np.min(np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)))
    if minsep < 1e-3:
        return 1e6
    return 0.0


def _objective(P, soft_beta, pen_scale):
    """Smooth surrogate: soft-min of vertex scores + convexity penalty (minimize)."""
    s = _vertex_scores(P)
    # soft-min via log-sum-exp: exp(-beta*min) ~ sum exp(-beta*s)
    neg = -soft_beta * s
    m = neg.max()
    softmin = (m - np.log(np.sum(np.exp(neg - m)))) / soft_beta
    return softmin + pen_scale * _convexity_penalty(P)


def _nm_minimize(f, x0, max_iter, ftol, xtol, timeout):
    """Nelder-Mead minimizer with explicit deadline guard."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    simplex = [x0.copy()]
    for i in range(1, n + 1):
        v = x0.copy()
        v[i - 1] += 0.05 if x0[i - 1] == 0 else 0.05 * x0[i - 1]
        simplex.append(v)
    simplex = np.array(simplex)
    fvals = np.array([f(p) for p in simplex])
    order = np.argsort(fvals)
    simplex = simplex[order]
    fvals = fvals[order]
    best_p = simplex[0].copy()
    best_f = fvals[0]
    it = 0
    start = time.time()
    while it < max_iter and (time.time() - start) < timeout:
        it += 1
        centroid = simplex[1:].mean(axis=0)
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    fvals[j] = f(simplex[j])
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_p = simplex[0].copy()
        if np.ptp(fvals) < ftol and np.ptp(simplex, axis=0).max() < xtol:
            break
    return best_p, best_f


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    hard_deadline = time.time() + 600.0

    # Start from the regular polygon (known valid, score 1.0).
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    P0 = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def to_p(x):
        return np.stack([x[0::2], x[1::2]], axis=1)

    def f_obj(x):
        return _objective(to_p(x), soft_beta=12.0, pen_scale=1.0)

    x0 = P0.ravel()

    best_P = P0.copy()
    best_raw = float(_vertex_scores(best_P).min())

    # Coarse multi-start Nelder-Mead to escape the regular-polygon local optimum.
    rng = np.random.default_rng(0)
    n_starts = 5
    budget = 480.0
    per_start = budget / n_starts
    for s in range(n_starts):
        if time.time() > hard_deadline - 60.0:
            break
        if s == 0:
            start = x0.copy()
        else:
            start = x0 + rng.normal(0.0, 0.06, size=2 * n)
        # keep it convex-ish by limiting perturbation; objective penalizes non-convex
        t_start = time.time()
        t_out = min(per_start, hard_deadline - t_start - 60.0)
        if t_out < 5.0:
            break
        p, fval = _nm_minimize(f_obj, start, max_iter=400, ftol=1e-7, xtol=1e-7, timeout=t_out)
        cand = to_p(p)
        if _convexity_penalty(cand) > 0:
            continue
        raw = float(_vertex_scores(cand).min())
        if raw > best_raw + 1e-9:
            best_raw = raw
            best_P = cand.copy()

    # High-precision polish pass on the best candidate.
    if time.time() < hard_deadline - 30.0:
        def f_polish(x):
            return _objective(to_p(x), soft_beta=40.0, pen_scale=1.0)
        t_start = time.time()
        t_out = min(240.0, hard_deadline - t_start - 30.0)
        p, _ = _nm_minimize(f_polish, best_P.ravel(), max_iter=800, ftol=1e-9, xtol=1e-9, timeout=t_out)
        cand = to_p(p)
        if _convexity_penalty(cand) == 0:
            raw = float(_vertex_scores(cand).min())
            if raw >= best_raw - 1e-9:
                best_P = cand.copy()
                best_raw = raw

    # Final validity guard: if anything broke, fall back to the regular polygon.
    if _convexity_penalty(best_P) != 0 or not np.all(np.isfinite(best_P)):
        return P0
    return best_P


# EVOLVE-BLOCK-END
