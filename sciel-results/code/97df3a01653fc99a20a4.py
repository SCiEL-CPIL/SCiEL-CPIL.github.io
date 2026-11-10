# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_indices(n=13):
    """Precompute all triangle index triples for n points."""
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

def _compute_ratio(points):
    """Compute A_min / A_hull ratio."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 1e-12:
            return 0.0
        # Vectorized triangle area computation
        n = len(points)
        idx = _triangle_indices(n)
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        # area = 0.5 * |cross(p2-p1, p3-p1)|
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = np.abs(cross) / 2.0
        a_min = np.min(areas)
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_objective(params, n=13, beta=50.0, idx=None):
    """
    Negative soft-min of triangle areas normalized by hull area.
    params: 26 values (13 x, 13 y)
    Uses log-sum-exp trick for smooth approximation to min.
    soft_min(x) = -logsumexp(-beta * x) / beta
    """
    if idx is None:
        idx = _triangle_indices(n)
    
    pts = params.reshape(n, 2)
    
    # Check validity
    if np.any(~np.isfinite(pts)):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
    except Exception:
        return 1e10
    
    if a_hull <= 1e-12:
        return 1e10
    
    # Triangle areas
    p1 = pts[idx[:, 0]]
    p2 = pts[idx[:, 1]]
    p3 = pts[idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) / 2.0
    
    # Soft min via log-sum-exp: soft_min = -logsumexp(-beta * areas) / beta
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    ratio = soft_min / a_hull
    return -ratio

def _objective_wrapper(params, beta, idx):
    return _soft_min_objective(params, 13, beta, idx)

def _grid_initialization(n=13, grid_size=6, rng=None):
    """
    Generate diverse initial configurations using a structured grid approach.
    Points are placed on a grid within the unit square, then normalized.
    This provides diverse starting points that avoid the regular polygon local minimum.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Grid-based: try different grid sizes and random selections
    for gs in [grid_size, grid_size + 1, grid_size + 2]:
        # Create grid points
        grid_x = np.linspace(0.1, 0.9, gs)
        grid_y = np.linspace(0.1, 0.9, gs)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
        
        # Randomly select n points from the grid
        for _ in range(3):
            perm = rng.permutation(len(grid_pts))
            selected = grid_pts[perm[:n]]
            # Add small random perturbation to break degeneracy
            noise = rng.normal(0, 0.02, (n, 2))
            selected = selected + noise
            # Normalize to unit area hull
            try:
                hull = ConvexHull(selected)
                if hull.volume > 1e-6:
                    inits.append(selected)
            except Exception:
                pass
    
    # Random points in unit square
    for _ in range(5):
        pts = rng.uniform(0.1, 0.9, (n, 2))
        try:
            hull = ConvexHull(pts)
            if hull.volume > 1e-6:
                inits.append(pts)
        except Exception:
            pass
    
    return inits

def _refine_with_gradient(pts, idx, deadline, rng):
    """
    Refine a point configuration using multi-scale gradient optimization.
    Returns the best (points, ratio) found.
    """
    n = len(pts)
    best_ratio = _compute_ratio(pts)
    best_pts = pts.copy()
    
    # Multi-scale beta schedule: start coarse, end fine
    betas = [10.0, 25.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts per beta
        for restart in range(2):
            if time.time() > deadline - 30:
                break
            
            if restart == 0:
                x0 = best_pts.flatten()
            else:
                # Small perturbation
                noise = rng.normal(0, 0.01, (n, 2))
                x0 = (best_pts + noise).flatten()
            
            try:
                result = minimize(
                    _objective_wrapper, x0,
                    args=(beta, idx),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
                )
                
                pts_out = result.x.reshape(n, 2)
                ratio = _compute_ratio(pts_out)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pts = pts_out.copy()
            except Exception:
                pass
    
    return best_pts, best_ratio

def _polish_with_nm(pts, deadline):
    """
    Final high-precision polish using Nelder-Mead with tight tolerances.
    """
    n = len(pts)
    
    def _nm_objective(p):
        pts_local = p.reshape(n, 2)
        return -_compute_ratio(pts_local)
    
    try:
        result = minimize(
            _nm_objective, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        pts_out = result.x.reshape(n, 2)
        ratio = _compute_ratio(pts_out)
        return pts_out, ratio
    except Exception:
        return pts, _compute_ratio(pts)

def heilbronn_convex13() -> np.ndarray:
    """
    Maximize A_min / A_hull for 13 points using smooth optimization
    with grid-based diverse initialization and multi-scale refinement.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    n = 13
    idx = _triangle_indices(n)
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = _grid_initialization(n, grid_size=6, rng=rng)
    
    best_ratio = 0.0
    best_points = None
    
    # Evaluate all initializations
    for init_pts in inits:
        ratio = _compute_ratio(init_pts)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = init_pts.copy()
    
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_ratio = _compute_ratio(best_points)
    
    # Refine the top initializations using gradient optimization
    # Sort by ratio and take top ones for refinement
    init_ratios = [( _compute_ratio(ip), ip) for ip in inits]
    init_ratios.sort(key=lambda x: -x[0])
    
    # Refine top 4 diverse initializations
    for _, init_pts in init_ratios[:4]:
        if time.time() > deadline - 60:
            break
        
        refined_pts, refined_ratio = _refine_with_gradient(init_pts, idx, deadline, rng)
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_points = refined_pts.copy()
    
    # Also refine the best found so far
    if time.time() < deadline - 60:
        refined_pts, refined_ratio = _refine_with_gradient(best_points, idx, deadline, rng)
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_points = refined_pts.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 20:
        polished_pts, polished_ratio = _polish_with_nm(best_points, deadline)
        if polished_ratio > best_ratio:
            best_ratio = polished_ratio
            best_points = polished_pts.copy()
    
    # Second polish pass if time allows
    if time.time() < deadline - 15:
        polished_pts, polished_ratio = _polish_with_nm(best_points, deadline)
        if polished_ratio > best_ratio:
            best_ratio = polished_ratio
            best_points = polished_pts.copy()
    
    # Final validation
    if best_points is None or best_points.shape != (13, 2):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points

# EVOLVE-BLOCK-END
