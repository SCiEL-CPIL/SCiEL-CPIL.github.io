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
    Strategy:
    1. Use a structural parametrization: points are defined by radii and angles.
       We assume a specific topology: some points on the hull, some inside.
       For 13 points, a common good configuration is 7 on the hull and 6 inside,
       or 8 on hull and 5 inside, etc.
       However, a robust approach is to optimize the coordinates directly but
       use a smooth objective function that is differentiable almost everywhere.
       
    2. The key insight from the inspirations is that Nelder-Mead is slow.
       L-BFGS-B is much faster for smooth objectives.
       The objective f(x) = -A_min / A_hull is non-smooth because of min().
       We use a soft-min approximation: 
       soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
       This is smooth and differentiable. As beta -> inf, it approaches min.
       
    3. We will perform a multi-start L-BFGS-B optimization:
       - Start from several diverse initial configurations (regular polygon, random, etc.)
       - Use a moderate beta for the soft-min to keep the gradient stable.
       - After L-BFGS-B converges, we can try to increase beta and re-optimize
         (homotopy method) to sharpen the minimum.
       
    4. Finally, we polish with a few steps of Nelder-Mead on the exact objective
       to ensure we are at the local optimum of the non-smooth function.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin for 630s limit
    
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

    def get_hull_area(pts):
        """Calculate convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0

    def smooth_objective(x, beta=50.0):
        """
        Smooth surrogate for global search.
        Minimizes - (soft_min_area / A_hull).
        soft_min_area = -1/beta * log(sum(exp(-beta * areas)))
        """
        pts = x.reshape((n, 2))
        areas = get_areas(pts)
        
        # Avoid numerical issues
        areas = np.maximum(areas, 1e-12)
        
        # Calculate hull area
        A_hull = get_hull_area(pts)
        if A_hull < 1e-12:
            return 1e6 # Penalty for degenerate hull
            
        # Soft minimum
        # To avoid overflow in exp, we use the log-sum-exp trick
        neg_beta_areas = -beta * areas
        max_neg = np.max(neg_beta_areas)
        lse = max_neg + np.log(np.sum(np.exp(neg_beta_areas - max_neg)))
        soft_min_area = -lse / beta
        
        # We want to maximize soft_min_area / A_hull
        # So we minimize - (soft_min_area / A_hull)
        return -(soft_min_area / A_hull)

    def exact_objective(x):
        """
        Exact objective for local polishing.
        Minimizes - (A_min / A_hull)
        """
        pts = x.reshape((n, 2))
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Define bounds for the search space. 
    # We search in a box [-1, 1] x [-1, 1].
    bounds = [(-1, 1)] * (2 * n)
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initial points
    rng = np.random.default_rng(seed=42)
    
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_regular = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(init_regular.flatten())
    
    # 2. Random uniform
    inits.append(rng.uniform(-1, 1, (n, 2)).flatten())
    
    # 3. Two concentric rings
    # 7 on outer, 6 on inner
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    init_rings = np.vstack([outer_pts, inner_pts])
    inits.append(init_rings.flatten())
    
    # 4. Hexagon + center + 6 more
    # 6 on hexagon, 1 center, 6 on a smaller hexagon rotated
    angles_hex = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hex_pts = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])
    center = np.array([[0, 0]])
    angles_inner_hex = angles_hex + np.pi/6
    inner_hex_pts = np.column_stack([0.5 * np.cos(angles_inner_hex), 0.5 * np.sin(angles_inner_hex)])
    init_hex = np.vstack([hex_pts, center, inner_hex_pts])
    inits.append(init_hex.flatten())
    
    # 5. Random with some structure (clumping)
    # Generate 3 clusters and assign points
    cluster_centers = rng.uniform(-0.8, 0.8, (3, 2))
    cluster_pts = []
    for i in range(n):
        c = cluster_centers[i % 3]
        offset = rng.normal(0, 0.2, 2)
        cluster_pts.append(c + offset)
    init_clusters = np.array(cluster_pts)
    inits.append(init_clusters.flatten())
    
    # Optimization Loop
    # We use L-BFGS-B on the smooth objective.
    # We start with a lower beta (smoother) and increase it (sharper).
    
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for x0 in inits:
        if time.time() > deadline - 60: # Leave time for polish
            break
            
        current_x = x0.copy()
        
        # Homotopy: increase beta step by step
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            try:
                # L-BFGS-B optimization
                res = minimize(
                    smooth_objective,
                    current_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < 1e5:
                    current_x = res.x
                    
            except Exception:
                pass
        
        # After L-BFGS-B, polish with Nelder-Mead on the exact objective
        if time.time() > deadline - 10:
            continue
            
        try:
            nm_res = minimize(
                exact_objective,
                current_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if nm_res.fun < 1e5:
                candidate_pts = nm_res.x.reshape((n, 2))
                candidate_score = compute_score(candidate_pts)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_pts.copy()
                    
        except Exception:
            pass

    # Fallback if nothing worked
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
