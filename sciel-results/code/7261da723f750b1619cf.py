"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_score(points):
    """Compute the packing score: min pair distance / 2, min distance to boundary."""
    n = len(points)
    radii = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - radii)
    if min_boundary < 0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_pair / 2.0, min_boundary)


def _soft_min_obj_and_grad(x, n, beta):
    """
    Soft-min objective using log-sum-exp for smoothness.
    We want to maximize t such that:
      t <= d_ij / 2 for all i < j
      t <= 1 - |p_i| for all i
    Equivalently, minimize -t.
    
    Using soft-min: t_soft = -(1/beta) * log(sum_i exp(-beta * a_i))
    where a_i are the constraints (d_ij/2 or 1-|p_i|).
    
    To make it efficient, we only consider "active" constraints near the minimum.
    But for a smooth surrogate, we can use all constraints with a temperature.
    
    However, 300 points give ~45000 pairs + 300 boundary = ~45300 constraints.
    Computing all exp(-beta * a_i) is expensive. Instead, we use the fact that
    only constraints near the minimum matter. We'll use a two-stage approach:
    1. Find approximate minimum constraint value
    2. Only include constraints within a window of the minimum
    """
    points = x.reshape(n, 2)
    radii = np.linalg.norm(points, axis=1)
    
    # Check validity
    if np.any(radii > 1.0):
        penalty = np.sum(np.maximum(0, radii - 1.0)) ** 2
        return 1e6 + penalty, np.zeros_like(x)
    
    # Pairwise distances via KDTree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    pair_dists = dists[:, 1]  # nearest neighbor distances
    
    # Boundary distances
    boundary_dists = 1.0 - radii
    
    # All constraint values
    # We'll use a soft-min over the nearest-neighbor pair distances and boundary distances
    # This is an approximation but captures the main constraints
    
    all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
    
    # Soft-min via log-sum-exp
    # t_soft = -(1/beta) * log(sum(exp(-beta * a_i)))
    # To avoid overflow, use max trick
    a = all_constraints
    a_max = np.max(a)
    # If all constraints are very negative (shouldn't happen if valid), handle
    if a_max < -1e10:
        return 1e6, np.zeros_like(x)
    
    exp_terms = np.exp(-beta * (a - a_max))
    log_sum = np.log(np.sum(exp_terms))
    t_soft = a_max - log_sum / beta
    
    # Gradient of t_soft w.r.t. points
    # d t_soft / d a_i = -exp(-beta*(a_i - a_max)) / sum(exp(-beta*(a_j - a_max)))
    weights = exp_terms / np.sum(exp_terms)
    
    # Now compute gradient of a_i w.r.t. points
    # a_i for i < n are pair_dists[i]/2 (nearest neighbor of point i)
    # a_i for i >= n are boundary_dists[i-n]
    
    grad = np.zeros_like(points)
    
    # For pair constraints: a_i = ||p_i - p_nearest(i)|| / 2
    # d a_i / d p_i = (p_i - p_nearest(i)) / (2 * ||p_i - p_nearest(i)||)
    # d a_i / d p_nearest(i) = -(p_i - p_nearest(i)) / (2 * ||p_i - p_nearest(i)||)
    
    # We need the nearest neighbor indices
    # Re-query to get indices
    dists2, idx2 = tree.query(points, k=2)
    nn_idx = idx2[:, 1]
    
    for i in range(n):
        j = nn_idx[i]
        d_ij = points[i] - points[j]
        norm_ij = np.linalg.norm(d_ij)
        if norm_ij > 1e-12:
            w = weights[i]
            grad[i] += w * d_ij / (2.0 * norm_ij)
            grad[j] -= w * d_ij / (2.0 * norm_ij)
    
    # For boundary constraints: a_i = 1 - ||p_i||
    # d a_i / d p_i = -p_i / ||p_i||
    for i in range(n):
        r = radii[i]
        if r > 1e-12:
            w = weights[n + i]
            grad[i] -= w * points[i] / r
    
    # Objective is -t_soft, so gradient is -grad
    obj = -t_soft
    return obj, (-grad).flatten()


def _generate_hex_lattice(n):
    """Generate hexagonal lattice points inside unit disk."""
    # Optimal hex spacing for n points in unit disk
    # Area per hex cell = s^2 * sqrt(3)/2
    # n * s^2 * sqrt(3)/2 <= pi
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0))) * 0.97
    
    k = int(2.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        y = j * s * np.sqrt(3.0) / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            if np.hypot(x, y) <= 1.0 - s / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n:
        # Fallback: add more points via spiral
        theta = np.linspace(0, 2 * np.pi * 20, n - len(points))
        r = np.linspace(0.1, 0.95, n - len(points))
        extra = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        points = np.vstack([points, extra])
    else:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')
        points = points[idx[:n]]
    
    return points[:n]


def _generate_ring_config(n):
    """Generate concentric ring configuration."""
    # Center point
    points = [[0.0, 0.0]]
    remaining = n - 1
    
    ring_radius = 0.15
    ring_count = 0
    while remaining > 0:
        # Number of points on this ring
        circumference = 2 * np.pi * ring_radius
        # Spacing ~ 2*r where r is target radius
        # Target r ~ 0.053
        target_spacing = 0.106
        count = max(1, int(circumference / target_spacing))
        count = min(count, remaining)
        
        if count > 0:
            theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
            ring_pts = np.column_stack([ring_radius * np.cos(theta), 
                                         ring_radius * np.sin(theta)])
            points.append(ring_pts)
            remaining -= count
        
        ring_radius += 0.08
        ring_count += 1
        if ring_radius > 0.9:
            break
    
    points = np.vstack(points)
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add more
        theta = np.linspace(0, 2 * np.pi, n - len(points))
        r = 0.95 * np.ones(n - len(points))
        extra = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        points = np.vstack([points, extra])
    
    return points[:n]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initial configurations
    candidates = []
    
    # Hexagonal lattice
    try:
        hex_pts = _generate_hex_lattice(N)
        if len(hex_pts) == N and np.all(np.isfinite(hex_pts)):
            candidates.append(hex_pts)
    except Exception:
        pass
    
    # Ring configuration
    try:
        ring_pts = _generate_ring_config(N)
        if len(ring_pts) == N and np.all(np.isfinite(ring_pts)):
            candidates.append(ring_pts)
    except Exception:
        pass
    
    # Random initialization
    try:
        np.random.seed(42)
        r = np.sqrt(np.random.rand(N)) * 0.95
        theta = np.random.rand(N) * 2 * np.pi
        rand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(rand_pts)
    except Exception:
        pass
    
    if not candidates:
        # Last resort
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.linspace(0, 0.95, N)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        try:
            score = _compute_score(cand)
            if score > best_score:
                best_score = score
                best_points = cand.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = candidates[0]
        best_score = _compute_score(best_points)
    
    # Local optimization using L-BFGS-B with soft-min objective
    # Use adaptive beta: start with lower beta (smoother) and increase
    
    beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        x0 = best_points.flatten()
        
        try:
            # Get initial objective and gradient
            obj, grad = _soft_min_obj_and_grad(x0, N, beta)
            
            if not np.isfinite(obj) or not np.all(np.isfinite(grad)):
                continue
            
            result = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.x is not None:
                new_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_points)):
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    restart_count = 0
    while time.time() < deadline - 15.0 and restart_count < 5:
        # Perturb best points
        noise_scale = 0.005
        x0 = best_points.flatten() + np.random.normal(0, noise_scale, size=N * 2)
        
        beta = 200.0
        try:
            result = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )
            
            if result.x is not None:
                new_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_points)):
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
        except Exception:
            pass
        
        restart_count += 1
    
    # Final validation
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0):
        mask = radii > 1.0
        best_points[mask] *= (1.0 - 1e-9) / radii[mask, np.newaxis]
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(best_points)):
        # Fallback
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.linspace(0, 0.95, N)
        best_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    return best_points
# EVOLVE-BLOCK-END
