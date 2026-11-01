"""Pack 50 equal circles in a unit disk, maximizing common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _get_radius(x):
    """Score: min over pairs of half-distance, and min over boundary gaps."""
    pts = x.reshape(-1, 2)
    d = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    r_boundary = d.min()
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    r_pair = dists[iu].min() / 2.0
    return min(r_pair, r_boundary)


def _objective(x, beta):
    """Soft-min surrogate: approximates -min(constraints)."""
    pts = x.reshape(-1, 2)
    d = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    pair = dists[iu] / 2.0
    vals = np.concatenate([pair, d])
    s = np.log(np.sum(np.exp(-beta * vals)))
    return s / beta


def _random_init(seed):
    rng = np.random.RandomState(seed)
    ang = rng.uniform(0, 2 * np.pi, N)
    rad = np.sqrt(rng.uniform(0, 1, N))
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]).ravel()


def _optimize_from(x0, betas, max_iter):
    best_x, best_r = x0.copy(), _get_radius(x0)
    for beta in betas:
        if _time_left() < 8.0:
            break
        res = minimize(_objective, best_x, args=(beta,),
                       method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12,
                                'gtol': 1e-10, 'maxls': 50})
        if res.x is not None and np.all(np.isfinite(res.x)):
            r = _get_radius(res.x)
            if r > best_r:
                best_r, best_x = r, res.x
    return best_x, best_r


def construct():
    t0 = time.time()
    deadline = t0 + 150.0

    # Stage 1: coarse-to-fine over several random restarts.
    betas_coarse = [10.0, 30.0, 80.0, 200.0, 500.0]
    best_x, best_r = None, -np.inf
    seed = 0
    while deadline - (time.time() - t0) > 40.0:
        x0 = _random_init(seed)
        x, r = _optimize_from(x0, betas_coarse, max_iter=200)
        if r > best_r:
            best_r, best_x = r, x
        seed += 1

    # Stage 2: high-precision polish on the best solution (sharpens soft-min to true min).
    betas_fine = [1000.0, 3000.0, 5000.0]
    if deadline - (time.time() - t0) > 15.0:
        x, r = _optimize_from(best_x, betas_fine, max_iter=300)
        if r > best_r:
            best_r, best_x = r, x

    # Final safety polish if time remains.
    if deadline - (time.time() - t0) > 6.0:
        x, r = _optimize_from(best_x, [5000.0], max_iter=200)
        if r > best_r:
            best_r, best_x = r, x

    return best_x.reshape(-1, 2)
