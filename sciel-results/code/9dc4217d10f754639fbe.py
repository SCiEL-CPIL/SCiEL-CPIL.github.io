"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 140.0  # safety margin before 150s kill


def _score(points):
    """Compute the common radius (min of half-pair-distances and container margins)."""
    n = len(points)
    if n == 0:
        return 0.0
    # Container constraint
    dists = np.hypot(points[:, 0], points[:, 1])
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return -1.0
    # Pair distances
    dx = points[:, 0:1] - points[0:1, 0]
    dy = points[:, 1:2] - points[0:1, 1]
    # Vectorized pairwise
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    pd = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(pd, np.inf)
    r_pair = np.min(pd) / 2.0
    return min(r_cont, r_pair)


def _build_ring_params(n_per_ring, initial_radii):
    """Build a points array from ring parameters: radii + angles per ring."""
    points = []
    for idx, (n, r) in enumerate(zip(n_per_ring, initial_radii)):
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points.extend(r * np.column_stack([np.cos(angles), np.sin(angles)]))
    return np.array(points)


def _neg_score(params, n_per_ring):
    """Negative score for minimization. params = [r0, r1, ..., a0_0, a0_1, ..., a1_0, ...]"""
    n_rings = len(n_per_ring)
    radii = params[:n_rings]
    # Angles: for ring i with n_i points, we have n_i angle offsets (relative to uniform)
    # Actually let's use: for each ring, a global rotation angle + individual perturbations
    # Simpler: for each ring, n_i angles (absolute)
    angle_start = n_rings
    points = []
    for idx, n in enumerate(n_per_ring):
        r = radii[idx]
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = params[angle_start:angle_start + n]
            angle_start += n
            pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
            points.extend(pts)
    pts = np.array(points)
    s = _score(pts)
    return -s if s > 0 else -s + 100.0  # penalty if invalid


def _neg_score_smooth(params, n_per_ring, beta=20.0):
    """Smooth version using log-sum-exp for the min."""
    n_rings = len(n_per_ring)
    radii = params[:n_rings]
    angle_start = n_rings
    points = []
    for idx, n in enumerate(n_per_ring):
        r = radii[idx]
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = params[angle_start:angle_start + n]
            angle_start += n
            pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
            points.extend(pts)
    pts = np.array(points)
    n = len(pts)
    if n == 0:
        return 100.0

    # Container
    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)

    # Pair distances - use soft min via log-sum-exp
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pd_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(pd_sq, np.inf)
    # soft min of pd/2
    m = np.min(pd_sq[pd_sq < np.inf])
    if m <= 0:
        return 100.0
    # log-sum-exp for soft min: -log(sum(exp(-beta * x))) / beta
    vals = pd_sq / 4.0  # (pd/2)^2, so we work with squared distances
    # Actually let's just use the exact min for reliability
    r_pair = np.sqrt(m) / 2.0

    r = min(r_cont, r_pair)
    if r < 0:
        return 100.0 - r
    return -r


def _local_refine(points, n_iter=200, step=0.001):
    """Simple gradient-free local refinement: try small perturbations."""
    best = points.copy()
    best_score = _score(best)
    n = len(best)
    for it in range(n_iter):
        improved = False
        for i in range(n):
            for axis in range(2):
                for direction in [1, -1]:
                    trial = best.copy()
                    trial[i, axis] += direction * step
                    s = _score(trial)
                    if s > best_score + 1e-12:
                        best = trial
                        best_score = s
                        improved = True
        if not improved:
            step *= 0.7
            if step < 1e-8:
                break
    return best


def construct():
    t0 = time.time()
    # Known structure: 1 + 6 + 12 + 18 + 13 = 50
    n_per_ring = [1, 6, 12, 18, 13]
    n_rings = len(n_per_ring)

    # Initial radii estimates (roughly evenly spaced)
    initial_radii = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

    # Build initial angles: uniform per ring
    n_angles = sum(max(0, n - 1) for n in n_per_ring)  # skip center
    # Actually for n=1 we don't need angles; for n>1 we need n angles
    angle_count = sum(n for n in n_per_ring if n > 1)
    x0 = np.zeros(n_rings + angle_count)
    x0[:n_rings] = initial_radii

    angle_idx = n_rings
    for n in n_per_ring:
        if n > 1:
            x0[angle_idx:angle_idx + n] = np.linspace(0, 2 * np.pi, n, endpoint=False)
            angle_idx += n

    # Optimize
    best_points = _build_ring_params(n_per_ring, initial_radii)
    best_score = _score(best_points)

    # Try L-BFGS-B
    try:
        result = minimize(
            _neg_score_smooth, x0, args=(n_per_ring,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < 0:
            x_opt = result.x
            # Reconstruct points
            radii = x_opt[:n_rings]
            angle_start = n_rings
            points = []
            for idx, n in enumerate(n_per_ring):
                r = radii[idx]
                if n == 1:
                    points.append([0.0, 0.0])
                else:
                    angles = x_opt[angle_start:angle_start + n]
                    angle_start += n
                    pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
                    points.extend(pts)
            pts = np.array(points)
            s = _score(pts)
            if s > best_score:
                best_points = pts
                best_score = s
    except Exception:
        pass

    # Local refinement
    if time.time() - t0 < _DEADLINE - 20:
        best_points = _local_refine(best_points, n_iter=150, step=0.005)
        best_score = _score(best_points)

    # Ensure validity
    if _score(best_points) < 0:
        # Fallback to simple ring packing
        radii = np.array([0.0, 0.15, 0.35, 0.55, 0.75])
        points = []
        for idx, n in enumerate(n_per_ring):
            r = radii[idx]
            if n == 1:
                points.append([0.0, 0.0])
            else:
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
                points.extend(pts)
        best_points = np.array(points)

    return best_points
# EVOLVE-BLOCK-END
