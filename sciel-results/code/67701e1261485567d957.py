"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _generate_points(r1, r2, theta1, theta2, dx, dy):
    """Generate 8 points: two concentric regular squares with independent radii, orientations, and 2D offset."""
    # Square 1: radius r1, orientation theta1, centered at origin
    angles1 = theta1 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    
    # Square 2: radius r2, orientation theta2, offset by (dx, dy)
    angles2 = theta2 + np.pi * np.array([0, 0.5, 1.0, 1.5])
    pts2 = np.column_stack([r2 * np.cos(angles2) + dx, r2 * np.sin(angles2) + dy])
    
    return np.vstack([pts1, pts2])

def _compute_score(pts):
    """Compute the score directly from 8 points. Returns the score value."""
    n = 8
    # Check distinctness and compute all pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-12:
                    return -np.inf
                dists[i, j] = d
    
    # Compute diameter (max distance)
    max_dist = np.max(dists)
    if max_dist < 1e-12:
        return -np.inf
    
    # Scale to diameter 2
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    # Compute product of distances over ordered pairs (i != j)
    log_product = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return -np.inf
                log_product += np.log(d)
    
    # Score = Delta / n^n
    log_score = log_product - n * np.log(n)
    return log_score

def _objective(params):
    """Negative log-score for minimization."""
    r1, r2, theta1, theta2, dx, dy = params
    if r1 < 1e-6 or r2 < 1e-6:
        return 1e10
    
    pts = _generate_points(r1, r2, theta1, theta2, dx, dy)
    log_score = _compute_score(pts)
    
    if np.isneginf(log_score) or np.isnan(log_score):
        return 1e10
    
    return -log_score

def construct():
    # 6 parameters: r1, r2, theta1, theta2, dx, dy
    bounds = [
        (0.1, 3.0),   # r1
        (0.1, 3.0),   # r2
        (0.0, np.pi / 2),  # theta1
        (0.0, np.pi / 2),  # theta2
        (-2.0, 2.0),  # dx
        (-2.0, 2.0),  # dy
    ]
    
    # Regular octagon: r1 = r2 = 1, theta1 = 0, theta2 = pi/4, dx = 0, dy = 0
    initial_guess = [1.0, 1.0, 0.0, np.pi / 4, 0.0, 0.0]
    
    # Run differential evolution with multiple seeds for robustness
    best_result = None
    best_fun = np.inf
    
    for seed in [42, 123, 456, 789, 101]:
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=seed,
                maxiter=500,
                tol=1e-12,
                polish=True,
                workers=1,
                popsize=15
            )
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is not None and best_result.fun < 1e9:
        best_params = best_result.x
    else:
        best_params = initial_guess
    
    # Local refinement using L-BFGS-B
    try:
        local_result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if local_result.fun < best_fun:
            best_params = local_result.x
    except Exception:
        pass
    
    pts = _generate_points(*best_params)
    
    # Scale to diameter 2
    n = 8
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            max_dist = max(max_dist, d)
    
    if max_dist < 1e-12:
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    
    scale = 2.0 / max_dist
    pts_scaled = pts * scale
    
    return [[float(pts_scaled[i][0]), float(pts_scaled[i][1])] for i in range(8)]
# EVOLVE-BLOCK-END
