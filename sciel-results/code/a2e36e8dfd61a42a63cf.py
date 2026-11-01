"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time


def _score(pts):
    """Compute the radius score: min of half-min-pair-dist and min-wall-dist."""
    n = len(pts)
    # Wall distance
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall)
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)


def _force_relaxation(pts, max_iter=500, dt=0.001, repulse_k=1.0, wall_k=1.0):
    """Simple force-based relaxation: repulsive forces between circles + wall constraints."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    # Initial estimate of spacing
    spacing = 1.0 / (int(np.ceil(np.sqrt(n))) + 0.5)
    for it in range(max_iter):
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < spacing * 3:
                    if dist < 1e-10:
                        d = np.array([1e-5, 0.0])
                        dist = 1e-5
                    f = repulse_k * (spacing - dist) / dist
                    forces[i] += f * d
                    forces[j] -= f * d
        # Wall repulsion
        for i in range(n):
            for axis in range(2):
                if pts[i, axis] > 0.5 - spacing:
                    forces[i, axis] -= wall_k * (pts[i, axis] - (0.5 - spacing))
                elif pts[i, axis] < -0.5 + spacing:
                    forces[i, axis] += wall_k * ((-0.5 + spacing) - pts[i, axis])
        # Update
        pts += dt * forces
        # Project back to square
        pts = np.clip(pts, -0.5 + 0.001, 0.5 - 0.001)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Initialize with a hexagonal-like pattern
    m = int(np.ceil(np.sqrt(N)))
    # Create a grid with some perturbation
    xs = np.linspace(-0.45, 0.45, m)
    ys = np.linspace(-0.45, 0.45, m)
    pts = np.array([[x, y] for y in ys for x in xs])[:N]
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    pts = pts + rng.normal(0, 0.001, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Phase 1: Force relaxation
    if time.time() < deadline - 30:
        relaxed = _force_relaxation(best_pts, max_iter=300, dt=0.0005, repulse_k=50.0, wall_k=100.0)
        if _score(relaxed) > best_score:
            best_pts = relaxed
            best_score = _score(best_pts)

    # Phase 2: L-BFGS-B optimization on the smooth surrogate
    # Objective: maximize min(distances) -> minimize negative of that
    # Use a soft-min approximation via log-sum-exp for smoothness

    def neg_soft_min(pts, lam=100.0):
        """Return negative soft-min of pair distances and wall distances."""
        n = len(pts)
        # Wall distances
        wall = 0.5 - np.max(np.abs(pts), axis=1)
        # Pair distances / 2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_half = dists / 2.0
        # Combine all constraints
        vals = np.concatenate([wall, pair_half[np.triu_indices(n, k=1)]])
        vals = np.maximum(vals, 1e-15)
        # Soft min via log-sum-exp
        m = np.min(vals)
        s = -m + np.log(np.sum(np.exp(lam * (vals - m)))) / lam
        return s  # We want to minimize this (it's approx -min(vals) + small)

    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalize going outside square
        out = np.maximum(0, np.abs(pts) - 0.5 + 0.001)
        penalty = 1000.0 * np.sum(out ** 2)
        return neg_soft_min(pts, lam=50.0) + penalty

    x0 = best_pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    if time.time() < deadline - 15:
        try:
            res = opt.minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            candidate = res.x.reshape(-1, 2)
            candidate = np.clip(candidate, -0.5, 0.5)
            sc = _score(candidate)
            if sc > best_score:
                best_pts = candidate
                best_score = sc
        except Exception:
            pass

    # Phase 3: High-precision polish with tighter tolerances
    if time.time() < deadline - 5:
        def precise_neg_min(x):
            pts = x.reshape(-1, 2)
            out = np.maximum(0, np.abs(pts) - 0.5 + 0.0001)
            penalty = 1e6 * np.sum(out ** 2)
            return neg_soft_min(pts, lam=200.0) + penalty

        x0 = best_pts.flatten()
        try:
            res = opt.minimize(
                precise_neg_min, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50}
            )
            candidate = res.x.reshape(-1, 2)
            candidate = np.clip(candidate, -0.5, 0.5)
            sc = _score(candidate)
            if sc > best_score:
                best_pts = candidate
                best_score = sc
        except Exception:
            pass

    # Final safety: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_pts
# EVOLVE-BLOCK-END
