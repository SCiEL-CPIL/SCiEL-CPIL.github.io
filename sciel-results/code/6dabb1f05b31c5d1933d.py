# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _precompute_indices(n):
    """Precompute indices for all triangles."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_score(points, tri_idx, hull_cache=None):
    """
    Compute Heilbronn score.
    Returns (score, areas, hull_area).
    """
    n = len(points)
    if n < 3:
        return 0.0, np.array([]), 0.0
        
    if not np.all(np.isfinite(points)):
        return 0.0, np.array([]), 0.0
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0, np.array([]), 0.0
        
    if A_hull <= 1e-12:
        return 0.0, np.array([]), 0.0
        
    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    A_min = np.min(areas)
    
    score = A_min / A_hull
    return score, areas, A_hull

def _smooth_objective(points, tri_idx, beta):
    """
    Smooth surrogate: -log(softmin(areas)) - log(hull_area)
    Equivalent to maximizing softmin(areas) / hull_area.
    As beta -> inf, softmin -> min.
    """
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
        if A_hull <= 1e-12:
            return 1e10
    except Exception:
        return 1e10

    p1 = points[tri_idx[:, 0]]
    p2 = points[tri_idx[:, 1]]
    p3 = points[tri_idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # We want to maximize this, so minimize its negative.
    # Objective to minimize: -softmin - log(A_hull)
    # softmin = -1/beta * log(sum(exp(-beta * areas)))
    # -softmin = 1/beta * log(sum(exp(-beta * areas)))
    
    # For numerical stability with large beta:
    # log_sum_exp(-beta * areas) = -beta * max(areas) + log(sum(exp(-beta * (areas - max(areas)))))
    # Actually we want min, so we look at -areas.
    # Let y = -areas. We want logsumexp(beta * y).
    # logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    
    z = -beta * areas
    z_max = np.max(z)
    if z_max > 500: # Overflow guard, though exp(-large) is small
         # This happens if areas are very small? No, z = -beta*areas. If areas > 0, z < 0.
         # If areas are close to 0, z is close to 0.
         # If areas are large, z is very negative.
         pass
         
    exp_sum = np.sum(np.exp(z - z_max))
    log_sum_exp = z_max + np.log(exp_sum)
    
    soft_min_val = -log_sum_exp / beta
    
    # We want to maximize soft_min_val / A_hull
    # Minimize -log(soft_min_val) - log(A_hull)
    # If soft_min_val <= 0, return penalty
    if soft_min_val <= 1e-12:
        return 1e10
        
    return -np.log(soft_min_val) - np.log(A_hull)

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin
    
    tri_idx = _precompute_indices(n)
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric: 1 center, 6 inner, 6 outer
    center = np.array([[0, 0]])
    ang_in = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ang_out = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    inner = np.column_stack([0.5 * np.cos(ang_in), 0.5 * np.sin(ang_in)])
    outer = np.column_stack([1.0 * np.cos(ang_out), 1.0 * np.sin(ang_out)])
    inits.append(np.vstack([center, inner, outer]))
    
    # 3. Grid-like
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    rng = np.random.default_rng(42)
    pts += rng.random((13, 2)) * 0.1
    inits.append(pts)
    
    # 4. Random
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 5. Fibonacci lattice (good for uniform distribution)
    phi = (1 + np.sqrt(5)) / 2
    fib_pts = []
    for i in range(n):
        r = np.sqrt(i / n)
        theta = 2 * np.pi * i / phi
        fib_pts.append([r * np.cos(theta), r * np.sin(theta)])
    inits.append(np.array(fib_pts))

    # Simulated Annealing Parameters
    T_start = 0.5
    T_end = 1e-4
    alpha = 0.995  # Cooling rate
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        current_pts = init_pts.copy()
        current_score, _, _ = _compute_score(current_pts, tri_idx)
        current_obj = _smooth_objective(current_pts, tri_idx, beta=20.0)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        T = T_start
        steps = 0
        max_steps = 20000  # Limit per restart to ensure we try all inits
        
        # We will increase beta over time to sharpen the minimum
        # Start with beta=10, end with beta=1000
        
        while T > T_end and steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Update beta: linearly increase from 10 to 1000
            progress = 1.0 - (T / T_start)
            beta = 10.0 + 990.0 * progress
            
            # Generate neighbor: Gaussian perturbation
            # Scale perturbation with T
            noise = rng.normal(0, 1, (n, 2)) * T
            new_pts = current_pts + noise
            
            new_obj = _smooth_objective(new_pts, tri_idx, beta=beta)
            
            # Acceptance criterion
            if new_obj < current_obj:
                current_pts = new_pts
                current_obj = new_obj
                new_score, _, _ = _compute_score(current_pts, tri_idx)
                if new_score > best_score:
                    best_score = new_score
                    best_points = current_pts.copy()
            else:
                # Metropolis criterion
                delta_obj = new_obj - current_obj
                if delta_obj < 0:
                    accept_prob = 1.0
                else:
                    # exp(-delta / T)
                    # If delta is huge, exp is 0
                    if delta_obj / T > 50:
                        accept_prob = 0.0
                    else:
                        accept_prob = np.exp(-delta_obj / T)
                        
                if rng.random() < accept_prob:
                    current_pts = new_pts
                    current_obj = new_obj
                    new_score, _, _ = _compute_score(current_pts, tri_idx)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = current_pts.copy()
                        
            T *= alpha
            
        # Final local polish with Nelder-Mead on the raw score for the best found in this restart
        # This helps snap to the exact min-geometry
        if time.time() < deadline - 30:
            def raw_obj(x):
                pts = x.reshape((n, 2))
                s, _, _ = _compute_score(pts, tri_idx)
                if s == 0.0:
                    return 1.0
                return -s
            
            from scipy.optimize import minimize
            try:
                res = minimize(raw_obj, current_pts.flatten(), method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                final_pts = res.x.reshape((n, 2))
                final_score, _, _ = _compute_score(final_pts, tri_idx)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
