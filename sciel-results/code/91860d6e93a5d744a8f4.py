import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0

    # Work in polar coords: x = [r1, theta1, r2, theta2, ..., r8, theta8]
    # r in [0.05, 3.0], theta in [0, 2*pi)

    def polar_to_cart(x):
        r = x[0::2]
        th = x[1::2]
        return np.column_stack([r * np.cos(th), r * np.sin(th)])

    def log_score_from_pts(pts):
        """Return log(Delta/n^n) where Delta = (prod unordered d)^2, scaled to diameter 2."""
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return -np.inf
        diffs = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diffs**2, axis=2)
        iu = np.triu_indices(n, k=1)
        d2u = d2[iu]
        if np.min(d2u) < 1e-18:
            return -np.inf
        du = np.sqrt(d2u)
        D = np.max(du)
        if D < 1e-12:
            return -np.inf
        # scaled distances = du * 2 / D
        # log product unordered = sum(log du) + 28 log 2 - 28 log D
        # Delta = (prod unordered)^2 => log Delta = 2 * (sum log du + 28 log 2 - 28 log D)
        # log score = log Delta - n log n = 2*(sum log du + 28 log 2 - 28 log D) - 8 log 8
        s = np.sum(np.log(du))
        return 2.0 * (s + 28.0 * math.log(2.0) - 28.0 * math.log(D) - 8.0 * math.log(8.0))

    def score_from_pts(pts):
        ls = log_score_from_pts(pts)
        if not np.isfinite(ls):
            return 0.0
        return math.exp(ls)

    def objective_polar(x):
        """Minimize negative log score (smooth)."""
        r = x[0::2]
        th = x[1::2]
        if np.any(r < 1e-3):
            return 1e10
        pts = r[:, None] * np.column_stack([np.cos(th), np.sin(th)])
        diffs = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diffs**2, axis=2)
        iu = np.triu_indices(n, k=1)
        d2u = d2[iu]
        if np.min(d2u) < 1e-14:
            return 1e10
        du = np.sqrt(d2u)
        # Smooth max via log-sum-exp
        log_d = np.log(du)
        m = np.max(log_d)
        D_smooth = m + np.log(np.sum(np.exp(log_d - m)))
        s = np.sum(log_d)
        # log score = 2*(s + 28 log 2 - 28 log D_smooth) - 8 log 8
        val = 2.0 * (s + 28.0 * math.log(2.0) - 28.0 * D_smooth - 8.0 * math.log(8.0))
        return -val

    def try_candidate(pts):
        nonlocal best_score, best_pts
        s = score_from_pts(pts)
        if s > best_score:
            best_score = s
            best_pts = np.array(pts, dtype=float)
        return s

    # Regular octagon (diameter 2 means radius 1)
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    reg_score = score_from_pts(reg_pts)
    best_score = reg_score
    best_pts = reg_pts.copy()

    # Candidate polar vectors
    def pts_to_polar(pts):
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        th = np.arctan2(pts[:, 1], pts[:, 0])
        th = np.mod(th, 2 * np.pi)
        x = np.zeros(2 * n)
        x[0::2] = r
        x[1::2] = th
        return x

    # Try structured configurations
    # Two concentric squares (radii r1, r2, rotated 45 deg)
    for r1 in np.linspace(0.3, 1.5, 13):
        for r2 in np.linspace(0.3, 1.5, 13):
            if time.time() > deadline - 30.0:
                break
            a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            a2 = a1 + np.pi/4
            r = np.concatenate([np.full(4, r1), np.full(4, r2)])
            th = np.concatenate([a1, a2])
            pts = r[:, None] * np.column_stack([np.cos(th), np.sin(th)])
            try_candidate(pts)

    # Regular octagon with radial perturbation (varying radii)
    np.random.seed(0)
    for trial in range(40):
        if time.time() > deadline - 30.0:
            break
        r = 1.0 + 0.4 * np.random.randn(n)
        r = np.clip(r, 0.2, 2.0)
        th = np.linspace(0, 2*np.pi, n, endpoint=False) + 0.15 * np.random.randn(n)
        pts = r[:, None] * np.column_stack([np.cos(th), np.sin(th)])
        try_candidate(pts)

    # Differential evolution in polar space
    bounds_polar = [(0.1, 3.0), (0, 2*np.pi)] * n

    def de_obj(x):
        if time.time() > deadline - 10.0:
            return 1e10
        return objective_polar(x)

    # Multiple DE restarts with different seeds
    for seed in [0, 1, 2, 3]:
        if time.time() > deadline - 15.0:
            break
        try:
            res = differential_evolution(
                de_obj, bounds_polar,
                seed=seed, maxiter=80, popsize=12,
                tol=1e-9, polish=False, init='latinhypercube'
            )
            if np.all(np.isfinite(res.x)):
                pts = polar_to_cart(res.x)
                try_candidate(pts)
        except Exception:
            pass

    # Polish best with L-BFGS-B in polar space
    if time.time() < deadline - 5.0:
        x0 = pts_to_polar(best_pts)
        # Unwrap theta: allow wider range for optimizer
        bounds_wide = [(0.05, 4.0), (-2*np.pi, 4*np.pi)] * n
        try:
            res = minimize(
                objective_polar, x0, method='L-BFGS-B',
                bounds=bounds_wide,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                pts = polar_to_cart(res.x)
                try_candidate(pts)
        except Exception:
            pass

    # Second polish round from multiple starting points (best + perturbed)
    if time.time() < deadline - 3.0:
        np.random.seed(7)
        starts = [best_pts.copy()]
        for _ in range(3):
            p = best_pts + 0.05 * np.random.randn(n, 2)
            starts.append(p)
        for p0 in starts:
            if time.time() > deadline - 2.0:
                break
            x0 = pts_to_polar(p0)
            try:
                res = minimize(
                    objective_polar, x0, method='L-BFGS-B',
                    bounds=bounds_wide,
                    options={'maxiter': 3000, 'ftol': 1e-19, 'gtol': 1e-15}
                )
                if np.all(np.isfinite(res.x)):
                    pts = polar_to_cart(res.x)
                    try_candidate(pts)
            except Exception:
                pass

    # Validate: distinct points, finite, coordinates in range
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, None, :] - best_pts[None, :, :]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    du = d[iu]
    if np.min(du) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
