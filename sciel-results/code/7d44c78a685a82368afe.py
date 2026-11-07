"""Second autocorrelation inequality: optimize 50 nonnegative step heights.

Strategy: explore scale-invariant shape space via a family of parametric
candidates (Gaussian, plateau, bimodal, tent, etc.) + multi-start L-BFGS-B
polish with a hard deadline guard.
"""
import numpy as np
import scipy.optimize
import time

N = 50
DEADLINE = 130.0  # safety margin before 150 s evaluator kill


def compute_value(h):
    """Exactly the evaluator's value, vectorized."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    xs = np.linspace(-0.5, 0.5, m + 2)
    w = xs[1] - xs[0]
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Vectorized:
    yi = y[:-1]
    yip1 = y[1:]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yip1 + yip1**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def compute_value_grad(h):
    """Compute value and gradient via finite differences (5-point stencil)."""
    h = np.asarray(h, dtype=np.float64)
    v = compute_value(h)
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(N):
        hp = h.copy()
        hm = h.copy()
        hp[i] += eps
        hm[i] -= eps
        grad[i] = (compute_value(hp) - compute_value(hm)) / (2 * eps)
    return v, grad


def _time_left():
    return DEADLINE - (time.perf_counter() - _START)


def _project(h):
    """Clip to nonnegativity and normalize (scale invariance doesn't matter
    for the ratio, but keeping max ~ 1 helps numerics)."""
    h = np.maximum(h, 0.0)
    mx = h.max()
    if mx > 0:
        h = h / mx
    return h


def _initial_candidates():
    """Generate a diverse set of initial shapes in [0,1]^50."""
    cands = []
    x = np.linspace(0, 1, N)

    # 1) Uniform
    cands.append(np.ones(N))

    # 2) Triangular / tent
    cands.append(1.0 - 2.0 * np.abs(x - 0.5))

    # 3) Plateau (center 50%)
    p = np.zeros(N)
    p[12:38] = 1.0
    cands.append(p)

    # 4) Narrow plateau (center 20%)
    p = np.zeros(N)
    p[20:30] = 1.0
    cands.append(p)

    # 5) Very narrow plateau (center 10%)
    p = np.zeros(N)
    p[23:27] = 1.0
    cands.append(p)

    # 6) Gaussian, sigma=0.15
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.15 ** 2)))

    # 7) Gaussian, sigma=0.08
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2)))

    # 8) Gaussian, sigma=0.25
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.25 ** 2)))

    # 9) Bimodal
    cands.append(np.exp(-((x - 0.3) ** 2) / (2 * 0.1 ** 2)) + np.exp(-((x - 0.7) ** 2) / (2 * 0.1 ** 2)))

    # 10) Raised cosine
    cands.append(0.5 * (1 + np.cos(2 * np.pi * x)))

    # 11) Power-law peak
    cands.append((1.0 - 2.0 * np.abs(x - 0.5)) ** 3)

    # 12) Two peaks narrow
    cands.append(np.exp(-((x - 0.25) ** 2) / (2 * 0.05 ** 2)) + np.exp(-((x - 0.75) ** 2) / (2 * 0.05 ** 2)))

    # 13) Uniform with small noise (deterministic seed)
    rng = np.random.RandomState(42)
    cands.append(0.5 + 0.5 * rng.rand(N))

    # 14) Smooth random (sine mix)
    cands.append(0.5 + 0.3 * np.sin(2 * np.pi * 2 * x) + 0.2 * np.sin(2 * np.pi * 3 * x + 1.0))

    # 15) Edge-concentrated
    cands.append(np.exp(-x / 0.2) + np.exp(-(1 - x) / 0.2))

    # 16) Wide Gaussian
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.35 ** 2)))

    # 17) Very wide Gaussian
    cands.append(np.exp(-((x - 0.5) ** 2) / (2 * 0.45 ** 2)))

    # 18) Step function (half on, half off)
    p = np.zeros(N)
    p[:25] = 1.0
    cands.append(p)

    # 19) Smooth step
    cands.append(1.0 / (1.0 + np.exp(-10 * (x - 0.5))))

    # 20) Parabolic
    cands.append(1.0 - 4.0 * (x - 0.5) ** 2)

    for i, c in enumerate(cands):
        cands[i] = _project(c)
    return cands


def _lbfgs_optimize(h0, start_time, maxiter=500, maxfun=8000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B on the shape with gradient."""
    def neg_value(h):
        v = compute_value(h)
        return -v

    def neg_value_grad(h):
        v, g = compute_value_grad(h)
        return -v, -g

    try:
        res = scipy.optimize.minimize(
            neg_value, h0, jac=neg_value_grad, method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': ftol, 'gtol': gtol}
        )
        return _project(res.x), -res.fun
    except Exception:
        return _project(h0), compute_value(_project(h0))


def construct():
    global _START
    _START = time.perf_counter()

    candidates = _initial_candidates()
    best_h = candidates[0].copy()
    best_val = compute_value(best_h)

    # First pass: quick L-BFGS-B from all candidates (short iterations)
    for i, h0 in enumerate(candidates):
        if _time_left() < 8.0:
            break
        h_opt, v = _lbfgs_optimize(h0, _START, maxiter=200, maxfun=3000, ftol=1e-12, gtol=1e-8)
        if v > best_val:
            best_val = v
            best_h = h_opt

    # Second pass: longer L-BFGS-B from top candidates
    # Re-rank candidates by their optimized values
    top_starts = []
    for i, h0 in enumerate(candidates):
        v = compute_value(h0)
        top_starts.append((v, h0))
    top_starts.sort(key=lambda x: -x[0])

    # Use top 10 initial shapes + current best for second pass
    second_starts = [h for _, h in top_starts[:10]] + [best_h]
    # Deduplicate
    unique_starts = [second_starts[0]]
    for s in second_starts[1:]:
        if np.linalg.norm(s - unique_starts[-1]) > 1e-6:
            unique_starts.append(s)

    for h0 in unique_starts:
        if _time_left() < 10.0:
            break
        h_opt, v = _lbfgs_optimize(h0, _START, maxiter=500, maxfun=8000, ftol=1e-14, gtol=1e-10)
        if v > best_val:
            best_val = v
            best_h = h_opt

    # Final polish: very tight tolerance from best solution
    if _time_left() > 5.0:
        h_opt, v = _lbfgs_optimize(best_h, _START, maxiter=1000, maxfun=15000, ftol=1e-16, gtol=1e-12)
        if v > best_val:
            best_val = v
            best_h = h_opt

    return list(best_h)


# Module-level start timestamp (set in construct, but also initialize here
# so _time_left works even if called early)
_START = time.perf_counter()
