"""Parametric concentric-ring packing: optimize ring radii + angles, then refine."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
# Known best packing structure: rings of 1,6,12,18,13 = 50 centers
RING_SIZES = [1, 6, 12, 18, 13]


def _compute_score(points):
    """Max common radius: min(min_pair_dist/2, min(1-|c_i|))."""
    if len(points) < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(points), k=1)
    min_pair = np.min(dists[iu]) / 2.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)


def _ring_to_points(params):
    """params: [r0, r1, r2, r3, r4, a0_0, a1_0..a1_5, a2_0..a2_11, a3_0..a3_17, a4_0..a4_12]
    First 5 = ring radii (ring 0 radius is ignored, center at origin).
    Angles are free. Returns (N,2) array."""
    points = np.zeros((N, 2))
    idx = 0
    # ring 0: single center at origin
    points[0] = [0.0, 0.0]
    idx = 1
    for ring in range(1, len(RING_SIZES)):
        rad = params[ring]
        size = RING_SIZES[ring]
        for k in range(size):
            ang = params[5 + idx]
            points[idx] = [rad * np.cos(ang), rad * np.sin(ang)]
            idx += 1
    return points


def _score_neg(params):
    """Objective = -score (to maximize)."""
    points = _ring_to_points(params)
    return -_compute_score(points)


def _ring_initial():
    """Initial guess: evenly spaced angles, radii scaled by ring index."""
    params = np.zeros(5 + (N - 1))
    # radii: spread rings roughly 0.3..0.85
    for ring in range(1, len(RING_SIZES)):
        params[ring] = 0.30 + 0.50 * ring / (len(RING_SIZES) - 1)
    idx = 0
    for ring in range(1, len(RING_SIZES)):
        size = RING_SIZES[ring]
        # even spacing with a per-ring offset
        offset = 0.0 if ring % 2 == 1 else np.pi / size
        for k in range(size):
            params[5 + idx] = offset + 2.0 * np.pi * k / size
            idx += 1
    return params


def _optimize_ring(params0, maxiter=800):
    """L-BFGS-B over the ring parameter space."""
    res = minimize(
        _score_neg,
        params0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30},
    )
    return res.x, -res.fun


def _free_refine(points, maxiter=400):
    """Refine in full (N,2) space with soft-min objective to lift local minima."""
    def obj_and_grad(x):
        beta = 150.0
        p = x.reshape(N, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, k=1)
        pair = dists[iu]
        norms = np.hypot(p[:, 0], p[:, 1])
        bound = 1.0 - norms
        all_d = np.concatenate([pair, bound])
        scaled = -beta * all_d
        m = np.max(scaled)
        e = np.exp(scaled - m)
        s = np.sum(e)
        w = e / s
        grad = np.zeros_like(x)
        for ii, (i, j) in enumerate(zip(iu[0], iu[1])):
            wi = w[ii]
            d = pair[ii]
            if d > 1e-15:
                direction = (p[i] - p[j]) / d
                grad[2*i] += wi * direction[0]
                grad[2*i+1] += wi * direction[1]
                grad[2*j] -= wi * direction[0]
                grad[2*j+1] -= wi * direction[1]
        for ii in range(N):
            wi = w[N + ii]
            n = norms[ii]
            if n > 1e-15:
                grad[2*ii] -= wi * p[ii, 0] / n
                grad[2*ii+1] -= wi * p[ii, 1] / n
        soft_min = -(m + np.log(s)) / beta
        return -soft_min, grad

    res = minimize(
        obj_and_grad,
        points.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20},
    )
    return res.x.reshape(N, 2)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    # Phase 1: optimize the parametric ring family
    best_params = _ring_initial()
    best_params, best_score = _optimize_ring(best_params)

    # Multi-restart in ring space with perturbations
    rng = np.random.default_rng(7)
    for restart in range(6):
        if time.time() > deadline - 60.0:
            break
        p0 = best_params.copy()
        # perturb angles and radii
        p0[1:5] += rng.normal(0, 0.05, size=4)
        p0[5:] += rng.normal(0, 0.3, size=p0[5:].shape)
        p0[1:5] = np.clip(p0[1:5], 0.1, 0.95)
        p1, s1 = _optimize_ring(p0, maxiter=500)
        if s1 > best_score:
            best_score = s1
            best_params = p1

    best_points = _ring_to_points(best_params)

    # Phase 2: free-space refinement to escape ring constraints
    if time.time() < deadline - 30.0:
        refined = _free_refine(best_points, maxiter=300)
        if np.all(np.isfinite(refined)):
            norms = np.hypot(refined[:, 0], refined[:, 1])
            if np.all(norms <= 1.0 + 1e-9):
                refined = np.clip(refined, -1.0, 1.0)
                # project slightly inside
                norms = np.hypot(refined[:, 0], refined[:, 1])
                mask = norms > 1.0
                if np.any(mask):
                    refined[mask] = refined[mask] / norms[mask, np.newaxis] * 0.999
                s = _compute_score(refined)
                if s > best_score:
                    best_score = s
                    best_points = refined

    # Phase 3: final high-precision polish in free space
    if time.time() < deadline - 10.0:
        polished = _free_refine(best_points, maxiter=300)
        if np.all(np.isfinite(polished)):
            norms = np.hypot(polished[:, 0], polished[:, 1])
            if np.all(norms <= 1.0 + 1e-9):
                polished = np.clip(polished, -1.0, 1.0)
                norms = np.hypot(polished[:, 0], polished[:, 1])
                mask = norms > 1.0
                if np.any(mask):
                    polished[mask] = polished[mask] / norms[mask, np.newaxis] * 0.999
                s = _compute_score(polished)
                if s > best_score:
                    best_score = s
                    best_points = polished

    # Ensure validity
    best_points = np.asarray(best_points, dtype=float)
    if not np.all(np.isfinite(best_points)):
        # fallback to ring solution
        best_points = _ring_to_points(best_params)
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999
    # ensure distinct centers
    return best_points
