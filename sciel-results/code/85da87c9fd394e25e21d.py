"""Program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius_deg(pts):
    """Compute covering radius in degrees. Returns inf if invalid."""
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = pts / norms[:, np.newaxis]
    
    # Check not in a closed hemisphere: origin should be inside convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    facets = hull.simplices
    max_r = 0.0
    
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Compute circumcenter of spherical triangle
        # The circumcenter is the point on the sphere equidistant from all three vertices
        # It's in the direction of the cross product of the two edge vectors
        e1 = b - a
        e2 = c - a
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        # Ensure n points to the same side as the triangle (toward centroid)
        centroid = (a + b + c) / 3.0
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-15:
            centroid = centroid / centroid_norm
            if np.dot(n, centroid) < 0:
                n = -n
        
        # Angular distance from circumcenter to each vertex
        d1 = np.arccos(np.clip(np.dot(n, a), -1.0, 1.0))
        d2 = np.arccos(np.clip(np.dot(n, b), -1.0, 1.0))
        d3 = np.arccos(np.clip(np.dot(n, c), -1.0, 1.0))
        r = max(d1, d2, d3)
        max_r = max(max_r, r)
    
    return np.degrees(max_r)


def _fast_covering_radius(pts):
    """Fast approximate covering radius using grid sampling."""
    # Create a grid of points on the sphere for quick evaluation
    # Use a Fibonacci lattice with ~10000 points
    n_grid = 10000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distance from each grid point to nearest configuration point
    # Use broadcasting: grid (n_grid, 3), pts (N, 3)
    # dist^2 = |g|^2 + |p|^2 - 2 g·p = 2 - 2 g·p (since both are unit vectors)
    dots = grid @ pts.T  # (n_grid, N)
    min_dots = np.max(dots, axis=1)  # max dot = min angular distance
    min_angles = np.arccos(np.clip(min_dots, -1.0, 1.0))
    return np.degrees(np.max(min_angles))


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_score = _covering_radius_deg(pts)
    best_pts = pts.copy()
    
    # Phase 1: Gradient-based optimization using the fast grid approximation
    # We minimize the maximum distance by using a smooth approximation
    # of the covering radius: the log-sum-exp of distances with high temperature
    
    def objective(x):
        """Objective function for optimization."""
        x = x / np.linalg.norm(x)
        # Reshape to (N, 3)
        p = x.reshape(N, 3)
        # Check for degeneracy
        if np.any(np.linalg.norm(p, axis=1) < 1e-10):
            return 1e10
        # Use fast covering radius
        return _fast_covering_radius(p)
    
    # Use L-BFGS-B on the sphere with parameterization
    # We'll use a simpler approach: iterative local optimization
    
    # Phase 2: Iterative local search with multiple restarts
    import random
    rng = np.random.RandomState(42)
    
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Try to improve each point
        for idx in range(N):
            if time.time() > deadline - 5:
                break
            
            # Generate random perturbation
            perturbation = rng.randn(3) * 0.05
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + perturbation
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            # Quick check with fast metric
            fast_new = _fast_covering_radius(new_pts)
            if fast_new < best_score:
                # Verify with exact metric
                exact_new = _covering_radius_deg(new_pts)
                if exact_new < best_score:
                    best_score = exact_new
                    best_pts = new_pts.copy()
                    pts = new_pts
    
    # Phase 3: Final refinement with smaller steps
    for iteration in range(500):
        if time.time() > deadline - 2:
            break
        
        idx = rng.randint(0, N - 1)
        perturbation = rng.randn(3) * 0.01
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + perturbation
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        fast_new = _fast_covering_radius(new_pts)
        if fast_new < best_score * 1.001:
            exact_new = _covering_radius_deg(new_pts)
            if exact_new < best_score:
                best_score = exact_new
                best_pts = new_pts.copy()
                pts = new_pts
    
    return best_pts
# EVOLVE-BLOCK-END
