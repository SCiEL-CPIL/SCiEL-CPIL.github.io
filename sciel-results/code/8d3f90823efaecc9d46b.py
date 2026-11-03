"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _objective(params):
    """
    Objective function for minimization: -min_triangle_area / hull_area
    """
    pts = params.reshape(N, 2)
    score = _compute_score(pts)
    if score <= 0:
        return 1e9
    return -score

def _halton_sequence(n, dim=2):
    """Generate n points in the Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        point = [halton(i, base) for base in [2, 3]]  # Bases for 2D
        points.append(point)
    return np.array(points)

def _generate_starts(rng):
    """Generate diverse initial points for multi-start optimization."""
    starts = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(p_reg)
    
    # 2. Halton sequence points (scaled to unit circle approx)
    halton_pts = _halton_sequence(N)
    # Center and scale
    halton_pts = halton_pts - 0.5
    # Scale to roughly fit in unit circle
    max_r = np.max(np.linalg.norm(halton_pts, axis=1))
    if max_r > 0:
        halton_pts = halton_pts / max_r
    starts.append(halton_pts)
    
    # 3. Jittered regular polygons
    for _ in range(5):
        noise = rng.normal(0, 0.1, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 4. Random convex-like starts
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.8, 1.2, N)
        p_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_rand)
        
    # 5. Grid-like starts (distorted)
    # Create a 4x4 grid, take 13 points
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    # Shuffle and take 13
    idx = rng.permutation(len(grid_pts))[:N]
    p_grid = grid_pts[idx]
    # Normalize to unit area approx
    p_grid = p_grid - p_grid.mean(axis=0)
    scale = np.max(np.linalg.norm(p_grid, axis=1))
    if scale > 0:
        p_grid = p_grid / scale
    starts.append(p_grid)
    
    return starts

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    starts = _generate_starts(rng)
    
    # Optimize each start
    for p0 in starts:
        if (time.time() - start_time) > time_limit - 5.0:
            break
            
        try:
            # Use Nelder-Mead for robustness against non-smoothness
            # maxiter is set high, but we rely on the time check in the main loop
            # or a tight gtol to stop early
            res = minimize(
                _objective, 
                p0.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.success or res.fun < 1e8:
                new_pts = res.x.reshape(N, 2)
                new_score = _compute_score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    
        except Exception:
            continue
            
    # If we have time, do a final polish on the best point found
    if best_pts is not None and (time.time() - start_time) < time_limit - 10.0:
        try:
            # Use Powell method for a finer polish, or Nelder-Mead with tighter tolerances
            res_polish = minimize(
                _objective, 
                best_pts.flatten(), 
                method='Powell', 
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            if res_polish.success or res_polish.fun < 1e8:
                polish_pts = res_polish.x.reshape(N, 2)
                polish_score = _compute_score(polish_pts)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_pts = polish_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

# EVOLVE-BLOCK-END
