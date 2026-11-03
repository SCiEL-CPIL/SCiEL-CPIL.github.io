"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area.
Uses smooth soft-min surrogate (log-sum-exp) with adaptive beta for gradient-based
optimization, plus a final exact-min Nelder-Mead polish pass."""
import time
import numpy as np
from scipy.optimize import minimize, NelderMead

N = 11
SQRT3 = np.sqrt(3.0)
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

# Precompute all C(11,3)=165 triple indices
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]


def _clamp_bary(b):
    """Clamp barycentric (u,v) to u>=0, v>=0, u+v<=1."""
    b = np.clip(b, 0.0, None)
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0
    if np.any(mask):
        b = b.copy()
        b[mask] = b[mask] / s[mask, None]
    return b


def _bary_to_cart(b):
    """Barycentric (u,v) -> Cartesian. point = u*V1 + v*V2."""
    x = b[:, 0] + 0.5 * b[:, 1]
    y = (SQRT3 / 2.0) * b[:, 1]
    return np.column_stack([x, y])


def _areas_from_cart(pts):
    """All 165 triangle areas for a set of N points (Cartesian)."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _min_area(pts):
    """Exact minimum triangle area."""
    return float(np.min(_areas_from_cart(pts)))


def _softmin_neg(x, beta):
    """Negative soft-min of triangle areas (smooth objective for minimization)."""
    b = _clamp_bary(x.reshape(-1, 2))
    pts = _bary_to_cart(b)
    areas = _areas_from_cart(pts)
    amin = np.min(areas)
    shifted = areas - amin
    # log-sum-exp for numerical stability
    m = np.min(-beta * shifted)
    lse = m + np.log(np.sum(np.exp(-beta * shifted - m)))
    soft_min = amin - lse / beta
    return -soft_min


def _exact_neg(x):
    """Negative exact min area (nonsmooth)."""
    b = _clamp_bary(x.reshape(-1, 2))
    pts = _bary_to_cart(b)
    return -_min_area(pts)


def _init_structured():
    """Structured initial barycentric config: vertices, edge midpoints, centroid,
    plus a few interior points on a coarse grid."""
    b = [
        [0.0, 0.0],    # V0
        [1.0, 0.0],    # V1
        [0.0, 1.0],    # V2
        [0.5, 0.0],    # mid V0-V1
        [0.5, 0.5],    # mid V1-V2
        [0.0, 0.5],    # mid V2-V0
        [1.0/3, 1.0/3],  # centroid
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.0, 0.25],
    ]
    return np.array(b[:N])


def _init_random(seed):
    rng = np.random.RandomState(seed)
    b = []
    for _ in range(2000):
        u = rng.uniform(0.0, 1.0)
        v = rng.uniform(0.0, 1.0 - u)
        b.append([u, v])
        if len(b) == N:
            break
    return np.array(b)


def _perturb(base, rng, scale):
    b = base + rng.normal(0.0, scale, size=base.shape)
    return _clamp_bary(b)


def _optimize_softmin(x0, beta_start, beta_end, maxiter, time_left_fn):
    """Optimize soft-min with L-BFGS-B, increasing beta over iterations."""
    n = len(x0)
    best_x = x0.copy()
    best_val = _softmin_neg(x0, beta_start)

    # Adaptive beta: linearly increase from beta_start to beta_end
    n_iter = maxiter
    for it in range(n_iter):
        if time_left_fn() < 5.0:
            break
        # beta increases with iteration
        frac = it / max(1, n_iter - 1)
        beta = beta_start + (beta_end - beta_start) * frac

        res = minimize(
            _softmin_neg,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        x0 = res.x
        val = _softmin_neg(x0, beta)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    return best_x


def _polish_exact(x0, maxiter, time_left_fn):
    """Nelder-Mead polish on exact min area."""
    best_x = x0.copy()
    best_val = _exact_neg(x0)

    res = minimize(
        _exact_neg,
        x0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12,
                 'adaptive': True},
    )
    x1 = res.x
    v1 = _exact_neg(x1)
    if v1 < best_val:
        best_x = x1
        best_val = v1
    return best_x


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    best_bary = None
    best_score = -np.inf

    def update(b):
        nonlocal best_bary, best_score
        pts = _bary_to_cart(b)
        s = _min_area(pts)
        if s > best_score:
            best_score = s
            best_bary = b.copy()
        return s

    # Initial structured start
    b0 = _init_structured()
    update(b0)

    # Phase 1: Soft-min optimization from structured start
    if time_left() > 10.0:
        x_opt = _optimize_softmin(b0.flatten(), beta_start=30.0, beta_end=300.0,
                                  maxiter=40, time_left_fn=time_left)
        b_opt = _clamp_bary(x_opt.reshape(-1, 2))
        update(b_opt)

    # Phase 2: Multi-restart with soft-min
    rng = np.random.RandomState(12345)
    restart = 0
    max_restarts = 200

    while restart < max_restarts and time_left() > 8.0:
        restart += 1

        # Choose start: mix of random, perturb-best, perturb-best-large
        r = restart
        if r % 5 == 0:
            b_start = _init_random(r * 7919)
        elif r % 2 == 0:
            scale = 0.03 + 0.03 * (1.0 - restart / max_restarts)
            b_start = _perturb(best_bary, rng, scale)
        else:
            scale = 0.08 + 0.05 * (1.0 - restart / max_restarts)
            b_start = _perturb(best_bary, rng, scale)

        # Soft-min optimization
        x_opt = _optimize_softmin(b_start.flatten(), beta_start=20.0, beta_end=200.0,
                                  maxiter=25, time_left_fn=time_left)
        b_opt = _clamp_bary(x_opt.reshape(-1, 2))
        update(b_opt)

    # Phase 3: Exact-min polish on best solution
    if time_left() > 5.0 and best_bary is not None:
        x_pol = _polish_exact(best_bary.flatten(), maxiter=500, time_left_fn=time_left)
        b_pol = _clamp_bary(x_pol.reshape(-1, 2))
        update(b_pol)

        # Another round of soft-min from polished solution
        if time_left() > 8.0:
            x_opt = _optimize_softmin(b_pol.flatten(), beta_start=50.0, beta_end=500.0,
                                      maxiter=30, time_left_fn=time_left)
            b_opt = _clamp_bary(x_opt.reshape(-1, 2))
            update(b_opt)

            # Final exact polish
            if time_left() > 5.0:
                x_pol2 = _polish_exact(b_opt.flatten(), maxiter=1000, time_left_fn=time_left)
                b_pol2 = _clamp_bary(x_pol2.reshape(-1, 2))
                update(b_pol2)

    # Fallback
    if best_bary is None:
        best_bary = _init_structured()

    return _bary_to_cart(best_bary)
