# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for validity: finite, non-collinear
    if not np.all(np.isfinite(points)):
        return 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-12:
        return 0.0
        
    # Vectorized area calculation for all triangles
    idx = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the Heilbronn ratio.
    Uses Simulated Annealing with L-BFGS-B local optimization on a smooth surrogate.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Pre-compute triangle indices for speed
    idx = np.array(list(itertools.combinations(range(n), 3)))
    i1, i2, i3 = idx[:, 0], idx[:, 1], idx[:, 2]
    
    def get_areas(pts):
        """Vectorized area calculation."""
        p1 = pts[i1]
        p2 = pts[i2]
        p3 = pts[i3]
        v1 = p2 - p1
        v2 = p3 - p1
        det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(det)

    def smooth_objective(x):
        """
        Smooth surrogate: -logsumexp(areas) / A_hull.
        Minimizing this maximizes the "effective" minimum area.
        """
        pts = x.reshape((n, 2))
        areas = get_areas(pts)
        
        # Log-sum-exp for stability
        max_a = np.max(areas)
        if max_a < 1e-12:
            return 1e6 # Huge penalty for degenerate
        lse = max_a + np.log(np.sum(np.exp(areas - max_a)))
        
        try:
            hull = ConvexHull(pts)
            A_hull = hull.volume
        except Exception:
            return 1e6
            
        if A_hull < 1e-12:
            return 1e6
            
        # We want to maximize min_area / hull_area.
        # logsumexp approximates max_area. To approximate min_area, we can use -logsumexp(-areas).
        # Let's use -logsumexp(-areas) / A_hull.
        # min(areas) ~ -logsumexp(-areas) (for large beta, but here we just use the smooth version)
        # Actually, -logsumexp(-areas) is a smooth lower bound on min(areas) that becomes tight as the spread decreases.
        # Let's minimize - ( -logsumexp(-areas) / A_hull ) = logsumexp(-areas) / A_hull.
        
        neg_areas = -areas
        max_neg = np.max(neg_areas)
        lse_neg = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        
        return lse_neg / A_hull

    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_regular)
    
    # 2. Concentric polygons (1 center, 6 inner, 6 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    init_concentric = np.vstack([center, inner_pts, outer_pts])
    inits.append(init_concentric)
    
    # 3. Grid-like jitter
    rng = np.random.default_rng(seed=42)
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.1
    inits.append(pts)

    # 4. Random points
    init_random = rng.random((n, 2)) * 2 - 1
    inits.append(init_random)

    # Simulated Annealing / Local Search Loop
    # We will run L-BFGS-B from multiple starting points, and use the results to seed further runs.
    # To save time, we limit the number of L-BFGS-B calls.
    
    candidates = [init.flatten() for init in inits]
    
    # We will process candidates in a queue.
    # If we have time, we add perturbed versions of the best found.
    
    max_lbfgs_calls = 30 # Rough estimate: 1-5s per call? Let's be conservative.
    lbfgs_calls_made = 0
    
    while candidates and time.time() < deadline and lbfgs_calls_made < max_lbfgs_calls:
        x0 = candidates.pop(0)
        
        # Run L-BFGS-B
        try:
            res = minimize(smooth_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success:
                final_pts = res.x.reshape((n, 2))
                final_score = compute_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
                    
                    # Add perturbed versions of the best point to the queue for further exploration
                    # This helps escape local minima
                    if time.time() < deadline - 60: # Safety margin for adding to queue
                        for _ in range(3):
                            noise = rng.normal(0, 0.05, (n, 2))
                            new_x = (final_pts + noise).flatten()
                            candidates.append(new_x)
                            
        except Exception:
            pass
            
        lbfgs_calls_made += 1

    # If no optimization succeeded, fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
