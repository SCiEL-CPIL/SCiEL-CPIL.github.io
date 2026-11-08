import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 145.0  # Safety margin before 150s

def _compute_radii_vec(x, y, W, H):
    """
    Vectorized computation of Voronoi radii.
    Returns an array of radii for N circles.
    """
    n = len(x)
    # 1. Distance to walls
    # r[i] = min(x[i], W-x[i], y[i], H-y[i])
    r = np.minimum(x, W - x)
    r = np.minimum(r, y)
    r = np.minimum(r, H - y)
    
    # 2. Distance to other circles
    # We need min_{j != i} 0.5 * ||p_i - p_j||
    # Create NxN distance matrix
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Set diagonal to infinity so it doesn't affect the min
    np.fill_diagonal(dist, np.inf)
    
    # Min distance to any other circle, halved
    min_dist_to_other = np.min(dist, axis=1) * 0.5
    
    # Final radius is min(wall_dist, min_dist_to_other)
    r = np.minimum(r, min_dist_to_other)
    
    return r

def _objective_vec(v, W, H):
    """Objective for gradient optimization: -sum(radii)."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _compute_radii_vec(x, y, W, H)
    return -np.sum(r)

def _validate(x, y, W, H, tol=1e-6):
    """Check validity. Note: Voronoi radii are always valid by construction 
    (disjoint interiors and inside box) if we use the computed radii.
    However, we check for numerical edge cases (r<=0)."""
    r = _compute_radii_vec(x, y, W, H)
    if np.any(r <= 0):
        return False, r
    # Strict check for overlaps just in case of floating point weirdness
    # Though Voronoi guarantees d >= r_i + r_j is NOT strictly true for 
    # "max radius" definition? 
    # Wait. If r_i is the max radius for i, and r_j is max radius for j.
    # Is it guaranteed that d_ij >= r_i + r_j?
    # Yes, because r_i <= d_ij/2 and r_j <= d_ij/2 => r_i + r_j <= d_ij.
    # So circles with Voronoi radii are always disjoint.
    # We just need to ensure they are inside the box.
    # r_i <= x_i, r_i <= W-x_i, etc. is guaranteed by the min operation.
    return True, r

def _sa_optimize(W, H, x0, y0, time_limit=5.0, T_init=0.1, T_end=0.001, cooling=0.95):
    """
    Simulated Annealing to find a good configuration.
    Returns best x, y found.
    """
    t_start = time.time()
    n = N
    x = x0.copy()
    y = y0.copy()
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = _compute_radii_vec(x, y, W, H)
    best_val = np.sum(best_r)
    
    current_r = best_r.copy()
    current_val = best_val
    
    T = T_init
    
    # Step size for random moves
    step_scale = 0.05
    
    while T > T_end:
        if time.time() - t_start > time_limit:
            break
            
        # Random perturbation
        # Move each particle randomly in a small box
        dx = np.random.uniform(-step_scale, step_scale, n)
        dy = np.random.uniform(-step_scale, step_scale, n)
        
        new_x = x + dx
        new_y = y + dy
        
        # Clip to box boundaries (centers must be inside)
        new_x = np.clip(new_x, 0.0, W)
        new_y = np.clip(new_y, 0.0, H)
        
        new_r = _compute_radii_vec(new_x, new_y, W, H)
        new_val = np.sum(new_r)
        
        delta = new_val - current_val
        
        # Accept if better, or with probability exp(delta / T)
        # Note: delta is negative if worse. We want to accept worse with prob exp(-|delta|/T)
        if delta > 0:
            accept = True
        else:
            # Avoid division by zero or underflow
            prob = np.exp(delta / T)
            accept = np.random.rand() < prob
            
        if accept:
            x = new_x
            y = new_y
            current_r = new_r
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_x = x.copy()
                best_y = y.copy()
                best_r = current_r.copy()
                
        T *= cooling
        
    return best_x, best_y, best_r, best_val

def _gradient_polish(x, y, W, H, time_limit=5.0):
    """
    Refine using L-BFGS-B.
    """
    t_start = time.time()
    n = N
    x0 = np.concatenate([x, y])
    bounds = [(0.0, W)] * n + [(0.0, H)] * n
    
    try:
        res = minimize(_objective_vec, x0, args=(W, H), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        
        x_new = res.x[:n]
        y_new = res.x[n:2*n]
        
        # Ensure we didn't drift out of bounds due to numerical issues
        x_new = np.clip(x_new, 0.0, W)
        y_new = np.clip(y_new, 0.0, H)
        
        r_new = _compute_radii_vec(x_new, y_new, W, H)
        val_new = np.sum(r_new)
        
        # Compare with old
        r_old = _compute_radii_vec(x, y, W, H)
        val_old = np.sum(r_old)
        
        if val_new > val_old:
            return x_new, y_new, r_new, val_new
        else:
            return x, y, r_old, val_old
            
    except Exception:
        r = _compute_radii_vec(x, y, W, H)
        return x, y, r, np.sum(r)

def construct():
    t0 = time.time()
    best_val = -1
    best_result = None
    
    # Aspect ratios to try (W + H = 2)
    # Symmetry: W in [1, 2) is unique, but we can try a range.
    # Perimeter 4 => 2(W+H)=4 => W+H=2.
    # If W=1, H=1 (Square).
    # If W=1.2, H=0.8.
    aspect_ratios = [
        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5,
        0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5
    ]
    
    # Grid configurations for initialization
    # 21 circles. 
    # 5x5 grid has 25 spots, we take 21.
    # 4x6 grid has 24 spots.
    # 3x7 grid has 21 spots exactly.
    grid_configs = [
        (5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)
    ]
    
    # We will iterate through aspect ratios.
    # For each aspect ratio, we will try a few initializations.
    # For each initialization, we run SA, then Gradient Polish.
    
    # To manage time, we distribute the budget.
    # Total time ~ 145s.
    # Let's do 20 aspect ratios. ~7s per aspect ratio.
    # Inside each aspect ratio, we can do 2-3 restarts.
    
    for ai, W in enumerate(aspect_ratios):
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Check time
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        # Try 2 different grid inits for this aspect ratio
        for gi in range(2):
            if time.time() - t0 > DEADLINE - 5.0:
                break
                
            cols, rows = grid_configs[(ai + gi) % len(grid_configs)]
            
            # Generate grid init
            xs = np.zeros(N)
            ys = np.zeros(N)
            k = 0
            for i in range(cols):
                for j in range(rows):
                    if k >= N:
                        break
                    cx = (i + 0.5) * W / cols
                    cy = (j + 0.5) * H / rows
                    # Small jitter to break symmetry
                    cx += (np.random.rand() - 0.5) * 0.01
                    cy += (np.random.rand() - 0.5) * 0.01
                    xs[k] = np.clip(cx, 0.01, W - 0.01)
                    ys[k] = np.clip(cy, 0.01, H - 0.01)
                    k += 1
            
            # 1. Simulated Annealing Phase
            # Time limit for SA: 3 seconds
            sa_time_limit = 3.0
            x_sa, y_sa, r_sa, val_sa = _sa_optimize(W, H, xs, ys, time_limit=sa_time_limit)
            
            # 2. Gradient Polish Phase
            # Time limit for Polish: 3 seconds
            pol_time_limit = 3.0
            x_pol, y_pol, r_pol, val_pol = _gradient_polish(x_sa, y_sa, W, H, time_limit=pol_time_limit)
            
            if val_pol > best_val:
                best_val = val_pol
                best_result = np.column_stack((x_pol, y_pol, r_pol))
                
    # If we still have time, do more random restarts on the best aspect ratio
    # We don't track which aspect ratio was best easily here, so we just try square (1.0) and 1.2
    if best_result is not None and time.time() - t0 < DEADLINE - 10.0:
        for W in [1.0, 1.2, 0.8]:
            H = 2.0 - W
            for seed in range(5):
                if time.time() - t0 > DEADLINE - 5.0:
                    break
                # Random init
                xs = np.random.uniform(0.1, W - 0.1, N)
                ys = np.random.uniform(0.1, H - 0.1, N)
                
                x_sa, y_sa, r_sa, val_sa = _sa_optimize(W, H, xs, ys, time_limit=2.0)
                x_pol, y_pol, r_pol, val_pol = _gradient_polish(x_sa, y_sa, W, H, time_limit=2.0)
                
                if val_pol > best_val:
                    best_val = val_pol
                    best_result = np.column_stack((x_pol, y_pol, r_pol))

    if best_result is not None:
        return best_result
    
    # Fallback
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])
