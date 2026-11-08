"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 14
DEADLINE = 140.0  # safety margin before 150s kill

def _score(points):
    """Compute min triangle area / convex hull area for given points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Convex hull area
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # in 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas
    # Triangle area = 0.5 * |cross(p2-p1, p3-p1)|
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    if min_area == float('inf') or min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _score_fast(points):
    """Vectorized score computation for speed."""
    n = len(points)
    if n < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorize triangle area computation
    # Use broadcasting for all triples
    # This can be memory intensive for large n, but n=14 is fine
    # Number of triples: C(14,3) = 364
    idx = np.array([[i, j, k] for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _normalize(points):
    """Normalize points to have convex hull area = 1."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area <= 1e-12:
            return points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.sqrt(1.0 / area)
        points = points * scale
    except Exception:
        pass
    return points


def _objective(x, n=N):
    """Objective function for minimization (negate score)."""
    points = x.reshape(n, 2)
    # Check for degenerate configurations
    if np.any(~np.isfinite(points)):
        return 1e10
    s = _score_fast(points)
    return -s


def construct():
    start_time = time.time()
    
    # Start with a reasonable initial configuration
    # Use a regular polygon-like distribution as starting point
    def _initial_points():
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Slightly perturb to break symmetry
        rng = np.random.RandomState(42)
        radii = 1.0 + 0.1 * rng.randn(N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pts
    
    best_points = _initial_points()
    best_score = _score_fast(best_points)
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(0)
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate random initial points
        if restart == 0:
            pts = best_points.copy()
        else:
            # Mix of random and perturbed best
            if restart % 2 == 0:
                pts = rng.randn(N, 2)
            else:
                pts = best_points + 0.3 * rng.randn(N, 2)
        
        # Normalize to unit hull area
        pts = _normalize(pts)
        
        # Local optimization using L-BFGS-B
        try:
            res = minimize(
                _objective,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            opt_pts = _normalize(opt_pts)
            opt_score = _score_fast(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
        
        # Also try a few more local optimizations with different perturbations
        for sub_restart in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbed = best_points + 0.1 * rng.randn(N, 2)
            perturbed = _normalize(perturbed)
            try:
                res = minimize(
                    _objective,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                opt_pts = res.x.reshape(N, 2)
                opt_pts = _normalize(opt_pts)
                opt_score = _score_fast(opt_pts)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Final polish with more iterations
    if time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_pts = res.x.reshape(N, 2)
            opt_pts = _normalize(opt_pts)
            opt_score = _score_fast(opt_pts)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Ensure we return a valid configuration
    final_score = _score_fast(best_points)
    if final_score <= 0:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
