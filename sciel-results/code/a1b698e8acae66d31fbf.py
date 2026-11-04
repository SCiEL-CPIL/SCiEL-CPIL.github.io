"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """
    Compute the score: minimum triangle area divided by convex hull area.
    Returns 0.0 for invalid/degenerate configurations.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute minimum triangle area over all C(13,3) = 286 triples
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                # Area of triangle with vertices points[i], points[j], points[k]
                # Using the determinant formula: 0.5 * |cross product|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if min_area < 0 or not np.isfinite(min_area):
        return 0.0
    
    return min_area / hull_area


def _objective(x):
    """
    Objective function for minimization.
    x is a flat array of 2*N coordinates.
    Returns negative score (so maximizing score = minimizing this).
    """
    points = x.reshape(N, 2)
    score = _compute_score(points)
    if score <= 0:
        return 1.0  # Large penalty for invalid configurations
    return -score


def construct():
    """
    Construct 13 points to maximize the minimum triangle area / convex hull area.
    Uses differential evolution with multiple restarts and a time budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_score = _compute_score(init_points)
    if init_score > best_score:
        best_score = init_score
        best_points = init_points.copy()
    
    # Define bounds for differential evolution
    # We'll work in a unit square and normalize later, or just use reasonable bounds
    bounds = [(-1, 1)] * (2 * N)
    
    # Run differential evolution with multiple restarts
    rng_seed = 42
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Create initial population
        np.random.seed(rng_seed + restart)
        
        # Use a variety of starting points
        if restart == 0:
            # Start near regular polygon
            x0 = init_points.flatten()
            # Add small noise
            x0 = x0 + np.random.randn(2 * N) * 0.1
        elif restart == 1:
            # Start with points on a circle with random angles
            angles = np.sort(np.random.uniform(0, 2 * np.pi, N))
            x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        elif restart == 2:
            # Start with random points in a disk
            r = np.sqrt(np.random.uniform(0, 1, N))
            theta = np.random.uniform(0, 2 * np.pi, N)
            x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()
        else:
            # Random start
            x0 = np.random.uniform(-1, 1, 2 * N)
        
        # Run differential evolution
        try:
            result = differential_evolution(
                _objective,
                bounds,
                x0=x0,
                seed=rng_seed + restart,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            
            if result.success or result.fun < -best_score:
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _compute_score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Final polish with scipy minimize if we have a good starting point
    if best_points is not None and time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        try:
            result = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < -best_score:
                candidate_points = result.x.reshape(N, 2)
                candidate_score = _compute_score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Normalize to unit area if we have a valid result
    if best_points is not None and best_score > 0:
        try:
            hull = ConvexHull(best_points)
            hull_area = hull.volume
            if hull_area > 1e-12:
                # Scale to unit area
                scale = 1.0 / np.sqrt(hull_area)
                best_points = best_points * scale
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular polygon
        best_points = init_points
    
    return best_points

# EVOLVE-BLOCK-END
