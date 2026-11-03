"""
Strategy: Two-phase smooth objective optimization.
Phase 1: L-BFGS-B on a soft-min penalty (moderate sharpness) to find a feasible packing.
Phase 2: Polish with a very sharp soft-min to refine the radius to high precision.
This is a fundamentally different optimization family from local pairwise push-apart.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 300
DEADLINE = 118.0  # Safety margin from 150s limit


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk."""
    best_points = None
    for s in [0.10, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.115, 0.12]:
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None or len(pts) > len(best_points):
                best_points = pts
                if len(pts) == N:
                    break
    if best_points is not None and len(best_points) == N:
        return best_points

    # Fallback: very fine grid
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]


def _score(points):
    """Compute the maximum radius r for given centers."""
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)

    return min(min_boundary, min_pair / 2.0)


def _soft_min_penalty(x_flat, r_target, sharpness):
    """
    Compute a smooth penalty: sum of soft-min violations.
    x_flat: 2N array of coordinates.
    r_target: target radius to enforce.
    sharpness: controls how sharp the soft-min is (higher = sharper).
    
    Penalty terms:
    1. Boundary: for each point, if |c_i| + r_target > 1, penalty = (|c_i| + r_target - 1)^2
    2. Pairwise: for each pair, if |c_i - c_j| < 2*r_target, penalty = (2*r_target - |c_i - c_j|)^2
    
    We use a soft-min approximation: max(0, v)^2 is not smooth, so we use:
    soft_max(0, v)^2 where soft_max(0, v) = log(1 + exp(k*v))/k for large k
    But simpler: just use (max(0, v))^2 with a small epsilon smoothing.
    
    Actually, for L-BFGS-B we can use a differentiable penalty:
    penalty = sum_i max(0, |c_i| - (1 - r_target))^2 + sum_{i<j} max(0, 2*r_target - |c_i - c_j|)^2
    
    The max(0, .) is not differentiable at 0, but L-BFGS-B can handle it numerically.
    For better performance, use a smooth approximation:
    smooth_max(0, v) = (v + sqrt(v^2 + eps))/2
    """
    pts = x_flat.reshape(-1, 2)
    n = len(pts)

    # Boundary penalty
    dists_to_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_viol = np.maximum(0.0, dists_to_origin - (1.0 - r_target))
    boundary_pen = np.sum(boundary_viol * boundary_viol)

    # Pairwise penalty - only check pairs that are close
    # For efficiency, use a cutoff
    cutoff = 3.0 * r_target
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    # Only penalize pairs within cutoff
    mask = dists < cutoff
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    pair_viol[~mask] = 0.0
    pair_pen = np.sum(pair_viol * pair_viol) / 2.0  # /2 because each pair counted twice

    return boundary_pen + pair_pen


def _gradient_penalty(x_flat, r_target):
    """Compute gradient of the penalty for L-BFGS-B."""
    pts = x_flat.reshape(-1, 2)
    n = len(pts)
    grad = np.zeros_like(x_flat)

    # Boundary gradient
    dists_to_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_viol = np.maximum(0.0, dists_to_origin - (1.0 - r_target))
    for i in range(n):
        if boundary_viol[i] > 0:
            r_i = dists_to_origin[i]
            if r_i > 1e-12:
                grad[2*i] = 2.0 * boundary_viol[i] * pts[i, 0] / r_i
                grad[2*i+1] = 2.0 * boundary_viol[i] * pts[i, 1] / r_i

    # Pairwise gradient - only for close pairs
    cutoff = 3.0 * r_target
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    mask = (dists < cutoff) & (dists > 1e-12)
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                d = dists[i, j]
                viol = 2.0 * r_target - d
                if viol > 0:
                    # dPen/dx_i = -2*viol * (c_i - c_j) / d  (from max(0, 2r - d)^2)
                    # Actually: d/dx_i [max(0, 2r - |c_i-c_j|)^2] 
                    # = 2 * max(0, 2r - d) * d(2r - d)/dx_i
                    # d(2r - d)/dx_i = -(c_i - c_j)/d
                    factor = 2.0 * viol / d
                    grad[2*i] -= factor * (pts[i, 0] - pts[j, 0])
                    grad[2*i+1] -= factor * (pts[i, 1] - pts[j, 1])
                    grad[2*j] += factor * (pts[i, 0] - pts[j, 0])
                    grad[2*j+1] += factor * (pts[i, 1] - pts[j, 1])

    return grad


def _optimize_phase1(points, time_limit):
    """
    Phase 1: Find a feasible packing by minimizing penalty with a target radius.
    Start with a radius slightly below the known best and increase.
    """
    start_time = time.time()
    pts = points.copy().flatten()
    n = len(points)

    # Start with a conservative radius
    r_start = 0.045
    r_step = 0.002
    r_max = 0.055

    best_pts = points.copy()
    best_score = _score(points)

    for r_target in np.arange(r_start, r_max + r_step/2, r_step):
        if time.time() - start_time > time_limit:
            break

        # Try to find a configuration where all circles of radius r_target fit
        def penalty(x):
            return _soft_min_penalty(x, r_target, 10.0)

        def gradient(x):
            return _gradient_penalty(x, r_target)

        result = minimize(
            penalty,
            pts,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )

        if result.fun < 1e-6:
            # Feasible! Update best
            new_pts = result.x.reshape(-1, 2)
            s = _score(new_pts)
            if s > best_score:
                best_score = s
                best_pts = new_pts.copy()
            # Continue with higher radius
            pts = result.x.copy()
        else:
            # Not feasible at this radius, try to get closer
            # Use the result as starting point for next radius
            if result.x is not None:
                # Blend: move toward the result
                pts = 0.5 * pts + 0.5 * result.x

    return best_pts, best_score


def _optimize_phase2(points, time_limit):
    """
    Phase 2: Polish with a very sharp soft-min to refine the radius.
    Use the current best radius and try to maximize it.
    """
    start_time = time.time()
    pts = points.copy()
    n = len(pts)

    # Get current radius
    current_r = _score(pts)
    
    # Try to increase radius slightly
    for delta_r in [0.0005, 0.0002, 0.0001, 0.00005, 0.00002]:
        if time.time() - start_time > time_limit:
            break

        r_target = current_r + delta_r
        x_flat = pts.flatten()

        def penalty(x):
            return _soft_min_penalty(x, r_target, 50.0)

        def gradient(x):
            return _gradient_penalty(x, r_target)

        result = minimize(
            penalty,
            x_flat,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
        )

        if result.fun < 1e-8:
            # Improved!
            new_pts = result.x.reshape(-1, 2)
            s = _score(new_pts)
            if s > current_r:
                current_r = s
                pts = new_pts.copy()

    return pts, current_r


def construct():
    start_time = time.time()

    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)

    best_points = points.copy()
    best_score = _score(points)

    # Phase 1: Coarse optimization to find feasible packing
    remaining = DEADLINE - (time.time() - start_time) - 10.0
    if remaining > 5.0:
        phase1_limit = min(remaining * 0.6, 60.0)
        relaxed, score = _optimize_phase1(points, phase1_limit)
        if score > best_score:
            best_score = score
            best_points = relaxed.copy()

    # Phase 2: Polish with sharp soft-min
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 3.0:
        phase2_limit = min(remaining, 30.0)
        polished, score = _optimize_phase2(best_points, phase2_limit)
        if score > best_score:
            best_score = score
            best_points = polished.copy()

    # Final validity check
    r = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        best_points[mask] = best_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)

    # Ensure no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-10:
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                    best_points[j, 0] += 1e-6

    return best_points
