"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area manually if ConvexHull fails
        # This can happen if points are collinear or nearly so
        n = len(points)
        if n < 3:
            return 0.0
        # Simple fallback: use the shoelace formula on the convex hull
        # For robustness, we'll just return a small positive value
        return 1e-10


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area


def _objective(x):
    """Objective function for minimization (we want to maximize score, so minimize negative score)."""
    points = x.reshape(-1, 2)
    s = _score(points)
    return -s


def _generate_initial_points(rng, n=N):
    """Generate initial points using a regular polygon with some perturbation."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Add some radial variation to avoid perfect symmetry which can be suboptimal
    radii = 1.0 + 0.1 * rng.random(n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Add small random perturbation
    points += 0.01 * rng.standard_normal((n, 2))
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_points = None
    
    # Multi-start Nelder-Mead optimization
    n_starts = 50
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        if start_idx == 0:
            # Start with regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
        else:
            # Generate random initial points with some structure
            # Mix of regular polygon and random perturbations
            if start_idx % 3 == 0:
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                radii = 0.8 + 0.4 * rng.random(N)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
            elif start_idx % 3 == 1:
                # Random points in unit square
                x0 = rng.random((N, 2)).flatten()
            else:
                # Regular polygon with larger perturbation
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1 * rng.standard_normal(N)
                radii = 1.0 + 0.2 * rng.standard_normal(N)
                x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        # Optimize using Nelder-Mead
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.success or True:  # Accept even if not fully converged
                points = result.x.reshape(-1, 2)
                s = _score(points)
                
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
        except Exception:
            continue
    
    # Final polish: run a few more iterations on the best solution found
    if best_points is not None and time.time() < deadline:
        x_best = best_points.flatten()
        try:
            result = minimize(
                _objective,
                x_best,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            points = result.x.reshape(-1, 2)
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
