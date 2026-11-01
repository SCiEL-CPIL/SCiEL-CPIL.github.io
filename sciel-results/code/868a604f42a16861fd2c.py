"""Rewritten packing solver for 300 equal circles in the unit disk."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
DEADLINE_MARGIN = 20  # seconds to keep before the 150 s wall-clock limit
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, 150.0 - DEADLINE_MARGIN - _elapsed())


def _compute_score(pts):
    """Return the largest common radius for the given centres."""
    if pts is None or len(pts) != N or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # distance from origin
    dist_origin = np.hypot(pts[:, 0], pts[:, 1])
    # if any point is outside or on boundary in a way that makes radius negative, invalid
    boundary = 1.0 - dist_origin
    if np.any(boundary <= 0):
        # still compute, but radius will be negative/zero; treat as invalid if any > 1
        if np.any(dist_origin >= 1.0):
            return 0.0
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_pair / 2.0, np.min(boundary))


def _hex_lattice_init(n):
    """Generate a hexagonal lattice clipped to the unit disk, then trim to n points."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Keep the n closest to origin for a balanced start
            order = np.argsort(radius, kind="stable")[:n]
            return points[order]
        spacing *= 0.98


def _ring_init(n, n_rings=12):
    """Generate points on concentric rings."""
    pts = []
    # center
    pts.append([0.0, 0.0])
    remaining = n - 1
    radii = np.linspace(0.15, 0.95, n_rings)
    counts = np.array([int(2 * np.pi * r * 50) for r in radii], dtype=int)
    counts = np.maximum(counts, 1)
    counts = counts * (remaining // counts.sum())
    # adjust to match exactly
    diff = remaining - counts.sum()
    idx = 0
    while diff != 0:
        if diff > 0:
            counts[idx % len(counts)] += 1
            diff -= 1
        else:
            if counts[idx % len(counts)] > 1:
                counts[idx % len(counts)] -= 1
                diff += 1
        idx += 1
    for r, cnt in zip(radii, counts):
        angles = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + np.random.uniform(0, 2 * np.pi / cnt)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        pts.extend(zip(x, y))
    pts = np.array(pts[:n])
    return pts


def _soft_min(distances, beta):
    """Soft minimum via log-sum-exp for positive values."""
    # distances are positive
    m = np.min(distances)
    s = np.sum(np.exp(-beta * (distances - m)))
    return m - np.log(s) / beta


def _objective_and_grad(x, beta):
    """
    Objective: negative of soft-min of (pairwise_dist/2, boundary_dist)
    We maximize r = min(...), so minimize -soft_min.
    x is a flat array of shape (N, 2).
    """
    pts = x.reshape(N, 2)
    # boundary distances
    dist_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - dist_origin
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_half = dist / 2.0
    # combine: we want soft min of all pair_half (off-diag) and boundary
    # For gradient, use log-sum-exp over all constraints
    # To avoid huge arrays, we can compute over a subset or all
    # For N=300, pair count is ~45k, which is manageable
    pair_vals = pair_half[~np.eye(N, dtype=bool)]
    all_vals = np.concatenate([pair_vals, boundary])
    # soft min
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    s = np.sum(exp_vals)
    obj = -(m - np.log(s) / beta)
    # gradient via chain rule: d obj / d x = -d soft_min / d x
    # d soft_min / d x = - (1/beta) * d log(sum exp(-beta(all_vals - m))) / d x + (dm/dx) * (1 - sum(exp.../s))
    # This is complex. Let's use a simpler approach: finite differences or a differentiable proxy.
    # For speed, let's use a differentiable objective: -min is not differentiable.
    # Alternative: minimize sum of squared violations + negative radius?
    # Let's use a barrier-like function.

    # Better approach: minimize -r where r is approximated by soft min.
    # Gradient of soft_min w.r.t. each constraint c_i:
    # d soft_min / d c_i = (1/beta) * exp(-beta(c_i - m)) / s
    # Then d c_i / d x for each constraint.

    # For boundary: c_i = 1 - |x_i|, dc/dx_i = -x_i / |x_i|
    # For pair: c_ij = |x_i - x_j| / 2, dc/dx_i = (x_i - x_j) / (2|x_i - x_j|), dc/dx_j = (x_j - x_i) / (2|x_i - x_j|)

    # Compute weights
    w = exp_vals / s / beta  # shape (len(all_vals),)

    # Accumulate gradient
    grad = np.zeros_like(pts)

    # Boundary contribution
    # indices for boundary are the last N of all_vals
    boundary_w = w[-N:]
    # dc/dx for boundary: -x_i / |x_i|
    with np.errstate(divide='ignore', invalid='ignore'):
        boundary_dir = -pts / dist_origin[:, np.newaxis]
        boundary_dir[dist_origin < 1e-10] = 0
    grad += boundary_w[:, np.newaxis] * boundary_dir

    # Pair contribution
    pair_w = w[:-N]
    # We need to map pair_w back to (i, j) indices
    # This is expensive. Let's approximate by only using a subset or use a different method.

    # For simplicity and speed, let's use a simpler objective:
    # Minimize sum(max(0, r - dist_ij/2)^2 + max(0, r - (1-|x_i|))^2) + lambda * (-r)
    # But r is not in the vector.

    # Let's switch to a simpler gradient-free or L-BFGS with finite differences.
    # Given time constraints, let's use L-BFGS-B with finite differences.

    return obj, None  # Return zero gradient; use finite differences


def _objective_simple(x):
    """
    Simple objective: -r where r is the minimum of all constraints.
    For optimization, we'll use a smoothed version.
    """
    pts = x.reshape(N, 2)
    dist_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - dist_origin
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_half = dist / 2.0
    r = min(np.min(pair_half), np.min(boundary))
    return -r


def construct():
    """
    Construct centres of 300 equal circles in the unit disk.
    Returns an array of shape (N, 2).
    """
    # Start with a hexagonal lattice
    x0 = _hex_lattice_init(N)
    best_pts = x0.copy()
    best_score = _compute_score(best_pts)

    # Try ring initialization as an alternative
    x0_ring = _ring_init(N)
    ring_score = _compute_score(x0_ring)
    if ring_score > best_score:
        best_pts = x0_ring.copy()
        best_score = ring_score

    # Optimization using L-BFGS-B with finite differences
    # We'll do multiple restarts with small perturbations

    # Use a simpler objective for L-BFGS-B
    def obj_fun(x):
        return _objective_simple(x)

    # Optimize
    best_x = best_pts.copy()
    best_obj = obj_fun(best_x.flatten())

    # Define a soft objective for better gradients
    def soft_obj(x, beta=100.0):
        pts = x.reshape(N, 2)
        dist_origin = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - dist_origin
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_half = dist / 2.0
        all_vals = np.concatenate([pair_half[~np.eye(N, dtype=bool)], boundary])
        m = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - m))
        s = np.sum(exp_vals)
        return -(m - np.log(s) / beta)

    # Optimization loop with time budget
    beta_schedule = [50, 200, 1000, 5000, 20000]
    for beta in beta_schedule:
        if _remaining() < 5.0:
            break

        # Use scipy minimize with L-BFGS-B
        try:
            res = minimize(
                soft_obj,
                best_x.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            if res.success or res.fun < best_obj:
                new_pts = res.x.reshape(N, 2)
                new_score = _compute_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts.copy()
                    best_score = new_score
                    best_obj = res.fun
        except Exception:
            pass

    # Final polish with a higher beta
    if _remaining() > 10.0:
        try:
            res = minimize(
                soft_obj,
                best_pts.flatten(),
                args=(100000.0,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            new_pts = res.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        except Exception:
            pass

    # Ensure the result is valid and within the unit disk
    dist_origin = np.hypot(best_pts[:, 0], best_pts[:, 1])
    # Clip points that might have drifted slightly outside
    mask = dist_origin < 1.0
    if not np.all(mask):
        # Scale back points that are outside
        for i in range(N):
            if dist_origin[i] >= 1.0:
                best_pts[i] *= (1.0 - 1e-6) / dist_origin[i]

    return best_pts
