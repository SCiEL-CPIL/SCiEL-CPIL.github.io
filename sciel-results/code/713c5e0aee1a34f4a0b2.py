import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _energy_and_grad(x):
    """
    x: shape (N, 3)
    Returns energy and gradient.
    Energy: Sum of inverse p-th power distances between all pairs + inverse p-th power distances to walls.
    Using a higher power p makes the energy more sensitive to the minimum distance,
    better approximating the max-min objective.
    """
    x = x.reshape(N, 3)
    p = 10  # Higher power for sharper focus on min distance
    
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    dist_sq_masked = np.where(mask, dist_sq, 1.0)
    
    # Energy from pairs: sum(1/d^p) = sum(1/d_sq^(p/2))
    # Let alpha = p/2 = 5
    alpha = p / 2.0
    e_pairs = np.sum(1.0 / (dist_sq_masked ** alpha))
    
    # Gradient of E_pairs
    # d/dx (1/d_sq^alpha) = -alpha * (1/d_sq^(alpha+1)) * 2*diff
    factor = -2.0 * alpha / (dist_sq_masked ** (alpha + 1))
    factor[np.eye(N, dtype=bool)] = 0.0
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad_pairs = np.sum(weighted_diff, axis=1)
    
    # Wall distances
    d_left = x + 0.5
    d_right = 0.5 - x
    
    eps = 1e-6
    d_left = np.maximum(d_left, eps)
    d_right = np.maximum(d_right, eps)
    
    # Wall energy: sum(1/d^p)
    e_walls = np.sum(1.0 / (d_left ** p)) + np.sum(1.0 / (d_right ** p))
    
    # Gradient of E_walls
    # d/dx (1/(x+0.5)^p) = -p/(x+0.5)^(p+1)
    # d/dx (1/(0.5-x)^p) = p/(0.5-x)^(p+1)
    grad_walls = -p / (d_left ** (p + 1)) + p / (d_right ** (p + 1))
    
    total_energy = e_pairs + e_walls
    total_grad = grad_pairs + grad_walls
    
    return total_energy, total_grad.flatten()

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
        
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1e10)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    min_wall_dist = np.min(0.5 - np.abs(pts))
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _bcc_initial_grid():
    """
    Generate a BCC lattice initialization.
    """
    # Generate BCC points in a box slightly larger than [-0.5, 0.5]
    # BCC unit cell size a.
    # We want roughly 100 points in volume 1.
    # Density of BCC is 2/a^3.
    # 100 = 2/a^3 => a^3 = 0.02 => a ~ 0.27
    # Let's try a = 0.25
    a = 0.25
    # Grid of cells
    n_cells = int(np.ceil(1.0 / a)) + 2
    # Start from -n_cells*a to n_cells*a
    start = -n_cells * a
    points = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                # Corner
                p1 = np.array([i*a, j*a, k*a])
                # Center
                p2 = p1 + np.array([a/2, a/2, a/2])
                points.append(p1)
                points.append(p2)
    points = np.array(points)
    # Filter points inside [-0.5, 0.5]
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    # If we have fewer, add random points (should not happen with this a)
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.4, 0.4, 3)
            points = np.vstack([points, p])
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    np.random.seed(42)
    
    # Try both grid and BCC initialization
    x_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    y_ticks = np.array([-0.375, -0.125, 0.125, 0.375])
    z_ticks = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    grid = np.array([(x, y, z) for x in x_ticks for y in y_ticks for z in z_ticks])
    base_grid = grid[:N]
    
    bcc_grid = _bcc_initial_grid()
    
    initial_guesses = [base_grid, bcc_grid]
    
    best_x = base_grid.copy()
    best_score = _calc_score(best_x)
    
    # Objective function for scipy
    def objective(x_flat):
        e, g = _energy_and_grad(x_flat)
        return e, g
        
    bounds = [(-0.5, 0.5)] * 3 * N
    
    jitter_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    
    for init_idx, base_init in enumerate(initial_guesses):
        for jitter in jitter_levels:
            if time.time() > deadline - 20:
                break
                
            # Create initial guess
            if jitter == 0.0:
                x0 = base_init.copy()
            else:
                x0 = base_init + np.random.uniform(-jitter, jitter, base_init.shape)
                x0 = np.clip(x0, -0.49, 0.49)
                
            # Run optimization
            try:
                # Use L-BFGS-B
                res = minimize(objective, x0.flatten(), method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if res.success:
                    new_x = res.x.reshape(N, 3)
                    new_x = np.clip(new_x, -0.5, 0.5)
                    new_score = _calc_score(new_x)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_x = new_x.copy()
                        
            except Exception as e:
                pass
            
        # If we have time, do a final fine-tuning on the best solution from this init
        if time.time() < deadline - 10:
            try:
                res_final = minimize(objective, best_x.flatten(), method='L-BFGS-B', jac=True,
                                     bounds=bounds,
                                     options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
                if res_final.success:
                    final_x = res_final.x.reshape(N, 3)
                    final_x = np.clip(final_x, -0.5, 0.5)
                    final_score = _calc_score(final_x)
                    if final_score > best_score:
                        best_score = final_score
                        best_x = final_x
            except Exception:
                pass

    return best_x
