"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair
    from scipy.spatial.distance import cdist
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Min distance to boundary
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_dist / 2.0, min_boundary)

def _smooth_objective(x):
    """
    Smooth objective function to minimize.
    Minimizes the sum of inverse squared distances to encourage separation.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)
    
    # Inverse squared distances (add epsilon to avoid division by zero)
    epsilon = 1e-12
    inv_dist_sq = 1.0 / (dist_sq + epsilon)
    
    # Sum of pairwise repulsions
    pair_repulsion = np.sum(inv_dist_sq) / 2.0
    
    # Boundary repulsions
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dist = np.minimum(dx, dy)
    
    # Ensure wall_dist is not negative (if point is outside, penalty should be huge)
    if np.any(wall_dist <= 0):
        return 1e6
    
    wall_repulsion = np.sum(1.0 / (wall_dist + epsilon)**2)
    
    return pair_repulsion + wall_repulsion

def _initial_grid():
    """Create a grid arrangement of N points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].astype(np.float64)

def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_pts = None
    best_radius = 0.0
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    rng = np.random.default_rng(42)
    
    # Initializations to try
    inits = []
    
    # 1. Standard Grid
    grid = _initial_grid()
    inits.append(grid)
    
    # 2. Perturbed Grids
    for _ in range(4):
        noise = rng.normal(0, 0.05, size=grid.shape)
        inits.append(np.clip(grid + noise, -0.5, 0.5))
        
    # 3. Random
    inits.append(rng.uniform(-0.4, 0.4, size=(N, 2)))

    candidates = []
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit - 30:
            break
            
        x0 = init.flatten().astype(np.float64)
        
        try:
            # Use L-BFGS-B for fast convergence on smooth problems
            res = minimize(
                _smooth_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            pts = res.x.reshape(-1, 2)
            pts = np.clip(pts, -0.5, 0.5)
            
            r = _compute_radius(pts)
            
            if r > best_radius:
                best_radius = r
                best_pts = pts.copy()
            
            # Keep top 3 candidates for refinement
            candidates.append((r, pts))
            
        except Exception:
            continue
            
    # Sort candidates by radius
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates with Nelder-Mead (derivative-free, robust to local minima)
    # Nelder-Mead is slower, so we only do it if we have time and for the best candidates
    if time.time() - start_time < time_limit - 20:
        for r, pts in candidates[:3]:
            if time.time() - start_time > time_limit - 10:
                break
                
            x0 = pts.flatten().astype(np.float64)
            try:
                # Nelder-Mead on the smooth objective
                res_nm = minimize(
                    _smooth_objective, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                pts_nm = res_nm.x.reshape(-1, 2)
                pts_nm = np.clip(pts_nm, -0.5, 0.5)
                
                r_nm = _compute_radius(pts_nm)
                
                if r_nm > best_radius:
                    best_radius = r_nm
                    best_pts = pts_nm.copy()
                    
            except Exception:
                continue

    # Fallback: if optimization failed or returned bad results, use the best initial or a safe grid
    if best_pts is None or best_radius < 0.05:
        best_pts = _initial_grid()
        best_radius = _compute_radius(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
