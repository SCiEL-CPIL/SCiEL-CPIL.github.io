"""Optimized logarithmic-energy minimizer for 150 points on the unit sphere."""
import time
import numpy as np

N = 150
DEADLINE = 132.0  # safety margin under the 150 s wall-clock kill
MIN_DIST = 1e-9


def _energy(pts):
    """Pairwise log(1/|x_i-x_j|) energy, lower is better."""
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, 1.0)
    return float(np.sum(np.log(1.0 / d)) / 2.0)


def _grad(pts):
    """Gradient of the energy in R^3 (radial component is spurious)."""
    diff = pts[:, None, :] - pts[None, :, :]          # (i-j)
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, 1.0)
    w = 1.0 / (d * d)
    w[np.eye(N, dtype=bool)] = 0.0
    return diff * w[:, :, None]


def _project(pts, g):
    """Project gradient onto the tangent space of the sphere."""
    return g - (g * pts).sum(axis=1, keepdims=True) * pts


def _optimize(pts, t_start, deadline):
    """Backtracking line-search gradient descent on the sphere."""
    pts = pts.copy()
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    lr = 1e-3
    for _ in range(3000):
        if time.time() > deadline:
            break
        g = _project(pts, _grad(pts))
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        step = g / ng * lr
        alpha = 1.0
        improved = False
        for _ in range(60):
            cand = pts + alpha * step
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            Ec = _energy(cand)
            if np.isfinite(Ec) and Ec < E - 1e-14:
                pts = cand
                E = Ec
                improved = True
                break
            alpha *= 0.5
        if improved:
            if E < best_E:
                best_E = E
                best_pts = pts.copy()
            lr = min(lr * 1.3, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    return best_pts, best_E


def _greedy_init(rng):
    """Greedy farthest-point sampling: a well-spread low-energy start."""
    pts = np.empty((N, 3))
    pts[0] = rng.normal(size=3)
    pts[0] /= np.linalg.norm(pts[0])
    min_d = np.full(N, np.inf)
    for i in range(1, N):
        d = np.linalg.norm(pts[None, :, :] - pts[:i, None, :], axis=2)
        min_d[i:] = np.minimum(min_d[i:], d.min(axis=1))
        j = np.argmax(min_d[i:]) + i
        pts[i] = pts[j]
    return pts


def _perturb(pts, rng, scale):
    """Add small random perturbations (tangent-ish) and renormalize."""
    noise = rng.normal(size=pts.shape) * scale
    out = pts + noise
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE
    rng = np.random.default_rng(12345)

    # Strong initial configuration via farthest-point sampling.
    init = _greedy_init(rng)
    best_pts, best_E = _optimize(init, t_start, deadline)

    # Restart with varied perturbations to escape local minima.
    scales = [0.05, 0.1, 0.2, 0.02, 0.3, 0.01]
    for s in scales:
        if time.time() > deadline:
            break
        cand = _perturb(best_pts, rng, s)
        p, e = _optimize(cand, t_start, deadline)
        if e < best_E:
            best_E = e
            best_pts = p

    # Final high-precision polish with a very small step.
    if time.time() < deadline - 5:
        p, e = _optimize(best_pts, t_start, deadline)
        if e < best_E:
            best_E = e
            best_pts = p

    # Final normalization and validity check.
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_pts)):
        return init
    return best_pts
