"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _params_to_points(params):
    """Convert parameters to 14 points on a deformed circle.
    
    params: array of shape (N,) representing angular offsets from uniform spacing.
    We use a Fourier-based deformation to maintain convexity.
    """
    n = N
    # Base angles: uniform spacing
    base_angles = 2 * np.pi * np.arange(n) / n
    
    # Apply angular deformation from params
    # params are small perturbations to angles
    angles = base_angles + params
    
    # Sort angles to maintain order (ensures convex polygon)
    angles = np.sort(angles)
    
    # Place points on unit circle at these angles
    x = np.cos(angles)
    y = np.sin(angles)
    
    return np.column_stack([x, y])


def _objective(params):
    """Objective function for differential evolution (maximize -> minimize negative)."""
    points = _params_to_points(params)
    s = _score(points)
    return -s


def construct():
    """Construct 14 points maximizing min triangle area / convex hull area."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with regular 14-gon
    best_points = None
    best_score = 0.0
    
    # Regular 14-gon on unit circle
    angles = 2 * np.pi * np.arange(N) / N
    regular_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(regular_points)
    best_points = regular_points.copy()
    
    # Try differential evolution with angular perturbations
    # Parameters: small angular offsets
    bounds = [(-0.5, 0.5)] * N
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            seed=42,
            polish=True,
            init='latinhypercube'
        )
        
        if time.time() < deadline - 10:
            candidate_points = _params_to_points(result.x)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead if time permits
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import minimize
            # Optimize directly in 2D space around best points
            def _obj_2d(flat_params):
                pts = flat_params.reshape(N, 2)
                return -_score(pts)
            
            # Multiple restarts from perturbed versions
            for restart in range(5):
                if time.time() > deadline - 15:
                    break
                
                if restart == 0:
                    start = best_points.flatten()
                else:
                    # Perturb best points
                    rng = np.random.RandomState(restart * 100 + 7)
                    start = best_points + rng.randn(N, 2) * 0.1
                    start = start.flatten()
                
                res = minimize(
                    _obj_2d,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                candidate_points = res.x.reshape(N, 2)
                candidate_score = _score(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points
        except Exception:
            pass
    
    # Final safety check
    if best_points is None:
        # Fallback: regular 14-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END
