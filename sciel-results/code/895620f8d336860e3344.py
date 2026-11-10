"""Program for 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill

def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
        
    # Pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_dist / 2.0, min_wall)

def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    # For N=200, a hexagonal grid with roughly 14x14 or similar is good.
    # We will try a few configurations and pick the one with the best radius.
    best_centres = None
    best_r = -1
    
    # Try different grid dimensions
    for n_rows in range(12, 18):
        for n_cols in range(12, 18):
            # Generate hex grid points
            # Hex grid spacing: horizontal = s, vertical = s * sqrt(3)/2
            # We want to fit n_cols * n_rows points roughly into 1x1 square.
            # Width = (n_cols - 1) * s + s (for offset) approx n_cols * s
            # Height = (n_rows - 1) * s * sqrt(3)/2
            
            # Let's define s based on width first
            s = 1.0 / n_cols
            height_needed = (n_rows - 1) * s * np.sqrt(3) / 2 + s
            if height_needed > 1.0:
                # Adjust s to fit height
                s = 1.0 / ((n_rows - 1) * np.sqrt(3) / 2 + 1)
            
            # Center the grid
            width = (n_cols - 1) * s + s
            height = (n_rows - 1) * s * np.sqrt(3) / 2 + s
            
            x_offset = -width / 2 + s / 2
            y_offset = -height / 2 + s * np.sqrt(3) / 2
            
            centres = []
            for row in range(n_rows):
                y = y_offset + row * s * np.sqrt(3) / 2
                for col in range(n_cols):
                    x = x_offset + col * s
                    if row % 2 != 0:
                        x += s / 2
                    centres.append((x, y))
            
            centres = np.array(centres)
            if len(centres) < N:
                continue
            
            centres = centres[:N]
            # Check bounds
            if np.any(centres < -0.5) or np.any(centres > 0.5):
                # Clip to valid region, though this might cause overlaps
                centres = np.clip(centres, -0.5, 0.5)
            
            r = _compute_radius(centres)
            if r > best_r:
                best_r = r
                best_centres = centres

    if best_centres is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        
    return best_centres

def _objective_and_grad(x, T):
    """
    Calculate the objective value (negative soft-min distance) and its gradient.
    We want to maximize the minimum distance (to walls and between particles).
    Let d_i be the constraints (distance to wall for i, distance/2 for pair i,j).
    We want to maximize min(d).
    Soft-min: -T * log(sum(exp(-d_i / T)))
    We minimize this function.
    """
    c = x.reshape(N, 2)
    
    # 1. Wall distances
    # d_wall_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    # This is a min of 4 terms. We can approximate the min with log-sum-exp too,
    # or just use the hard min for the wall term in the gradient if we are careful.
    # However, for a smooth optimization, it's better to include walls in the soft-min.
    
    # Let's create a vector of all "distances" to constraints.
    # Walls: 4N distances.
    # Pairs: N(N-1)/2 distances.
    # Total constraints: 4N + N(N-1)/2. For N=200, this is 800 + 19900 = 20700.
    # This is manageable for vectorized operations.
    
    # Wall distances
    wx = 0.5 - c[:, 0]
    wx_neg = 0.5 + c[:, 0]
    wy = 0.5 - c[:, 1]
    wy_neg = 0.5 + c[:, 1]
    
    wall_dists = np.concatenate([wx, wx_neg, wy, wy_neg])
    
    # Pairwise distances
    # Calculate all pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    # The radius constraint is d/2
    pair_dists_half = pair_dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([wall_dists, pair_dists_half])
    
    # Clip small distances to avoid log(0) or exp overflow issues
    all_dists = np.clip(all_dists, 1e-10, None)
    
    # Soft-min calculation
    # val = -T * log(sum(exp(-d / T)))
    # grad = sum(exp(-d/T) * grad_d) / sum(exp(-d/T))
    
    exp_vals = np.exp(-all_dists / T)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        # Should not happen with clipping
        return 1e10, np.zeros_like(x)
        
    obj_val = -T * (np.log(sum_exp))
    # Note: The constant terms in log-sum-exp don't matter for optimization, 
    # but we want the value to be comparable. 
    # Actually, -T * log(sum(exp(-d/T))) is a lower bound on -min(d).
    # As T -> 0, it approaches -min(d).
    
    # Gradient calculation
    # grad_x = d(obj)/dx = - d/dx [ -T log(sum exp(-d/T)) ]
    #        = T * [ 1/sum(exp) * sum( exp(-d/T) * d(-d/T)/dx ) ]
    #        = T * [ 1/sum(exp) * sum( exp(-d/T) * (-1/T) * d(-d)/dx ) ]
    #        = T * [ 1/sum(exp) * sum( exp(-d/T) * (1/T) * d(d)/dx ) ]
    #        = [ 1/sum(exp) * sum( exp(-d/T) * d(d)/dx ) ]
    #        = sum( w_k * d(d_k)/dx ) where w_k = exp(-d_k/T) / sum_exp
    
    grad = np.zeros_like(c)
    
    # Weights
    weights = exp_vals / sum_exp
    
    # 1. Wall contributions
    # d(wx)/dx = -1, d(wx_neg)/dx = 1, etc.
    # Indices 0..N-1: wx = 0.5 - x. d/dx = -1
    # Indices N..2N-1: wx_neg = 0.5 + x. d/dx = 1
    # Indices 2N..3N-1: wy = 0.5 - y. d/dy = -1
    # Indices 3N..4N-1: wy_neg = 0.5 + y. d/dy = 1
    
    # Sum of weights * derivatives
    # grad_x from walls:
    # sum(w[0:N] * (-1)) + sum(w[N:2N] * 1)
    grad[:, 0] -= np.sum(weights[0:N])
    grad[:, 0] += np.sum(weights[N:2*N])
    
    # grad_y from walls:
    grad[:, 1] -= np.sum(weights[2*N:3*N])
    grad[:, 1] += np.sum(weights[3*N:4*N])
    
    # 2. Pairwise contributions
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # d(d_ij)/dx_j = (x_j - x_i) / d_ij
    
    # We have pair_dists_half in the array.
    # Let's re-calculate the full distances for gradient direction to be safe, 
    # or use the stored ones.
    # pair_dists is the full distance. pair_dists_half is d/2.
    # d(d_half)/dx = 0.5 * d(d)/dx = 0.5 * (x_i - x_j)/d = (x_i - x_j)/(2*d) = (x_i - x_j) / (2 * 2 * d_half) ? No.
    # d_half = d/2.
    # d(d_half)/dx_i = 0.5 * (x_i - x_j)/d = 0.5 * (x_i - x_j)/(2*d_half) = (x_i - x_j)/(4*d_half).
    
    # Let's use the full distance 'dists' for the direction vector normalization.
    # d(d_half)/dx_i = (x_i - x_j) / (2 * dists[i,j])
    
    # We need to map the weights of pair_dists_half back to the pairs.
    # weights[4N:] correspond to the pairs in iu order.
    pair_weights = weights[4*N:]
    
    # Vectorized update for pairs
    # iu[0] contains row indices, iu[1] contains col indices.
    rows = iu[0]
    cols = iu[1]
    
    # Get distances for these pairs
    pair_d = dists[rows, cols]
    
    # Calculate direction vectors
    # diff = c[rows] - c[cols]
    diffs = c[rows] - c[cols]
    
    # Gradient contribution for particle i (rows):
    # grad_i += sum over j of w_ij * (x_i - x_j) / (2 * d_ij)
    # grad_j -= sum over i of w_ij * (x_i - x_j) / (2 * d_ij)
    
    # We need to accumulate these.
    # Using np.add.at for unbuffered in-place addition
    
    # Coefficients: w / (2 * d)
    coeffs = pair_weights / (2.0 * pair_d)
    
    # Weighted differences
    weighted_diffs = coeffs[:, np.newaxis] * diffs
    
    # Add to rows
    np.add.at(grad, rows, weighted_diffs)
    # Subtract from cols
    np.add.at(grad, cols, -weighted_diffs)
    
    return obj_val, grad.flatten()

def _optimize_l_bfgs_b(centres, time_budget):
    """
    Optimize using L-BFGS-B with a soft-min objective.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    x0 = centres.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Schedule for Temperature T
    # Start with a relatively high T to smooth the landscape, then decrease.
    # T is in units of distance. Radius is ~0.036.
    # If T is too high, the gradient is weak. If too low, it's noisy.
    
    # We will run L-BFGS-B in stages with decreasing T.
    current_x = x0.copy()
    
    # Stage 1: Coarse optimization
    T1 = 0.05
    # Stage 2: Medium
    T2 = 0.02
    # Stage 3: Fine
    T3 = 0.005
    # Stage 4: Polish
    T4 = 0.001
    
    stages = [
        (T1, 100, 5000),
        (T2, 200, 10000),
        (T3, 300, 15000),
        (T4, 500, 20000)
    ]
    
    for T, maxiter, maxfun in stages:
        if time.time() > deadline:
            break
            
        # Define objective and gradient for this T
        def obj(x):
            val, _ = _objective_and_grad(x, T)
            return val
        
        def grad(x):
            _, g = _objective_and_grad(x, T)
            return g
        
        try:
            result = minimize(
                obj, current_x,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                current_x = result.x
        except Exception:
            pass

    return current_x.reshape(N, 2)

def _polish(centres, time_budget):
    """
    Local search to fix any remaining overlaps or improve the minimum distance.
    This uses a simple hill-climbing approach on the exact min-radius objective.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    step_size = 0.01
    min_step = 1e-6
    
    while time.time() < deadline:
        # Find the bottleneck
        c = best_centres
        wall_dists = np.min(np.stack([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1]
        ], axis=1), axis=1)
        
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        
        # The radius is min(min_wall, min_pair/2)
        # We want to increase the minimum of these.
        # Identify the particle(s) involved in the minimum constraint.
        
        min_wall_val = np.min(wall_dists)
        min_pair_val = np.min(dists) / 2.0
        
        if min_wall_val < min_pair_val:
            # Bottleneck is a wall
            idx = np.argmin(wall_dists)
            # Try moving this particle away from the wall
            # Determine which wall
            x, y = c[idx]
            if x > 0:
                dx, dy = -1, 0
            elif x < 0:
                dx, dy = 1, 0
            elif y > 0:
                dx, dy = 0, -1
            else:
                dx, dy = 0, 1
            
            # Try a few perturbations
            improved = False
            for _ in range(20):
                angle = np.random.uniform(0, 2*np.pi)
                # Bias towards the direction away from wall
                base_angle = np.arctan2(dy, dx)
                angle = base_angle + np.random.uniform(-np.pi/3, np.pi/3)
                
                new_c = best_centres.copy()
                new_c[idx, 0] += step_size * np.cos(angle)
                new_c[idx, 1] += step_size * np.sin(angle)
                
                if np.any(new_c < -0.5) or np.any(new_c > 0.5):
                    continue
                    
                new_r = _compute_radius(new_c)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_c
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.95
                if step_size < min_step:
                    break
        else:
            # Bottleneck is a pair
            iu = np.argwhere(dists == np.min(dists))
            if len(iu) == 0:
                break
            i, j = iu[0]
            
            # Move i and j apart
            diff = best_centres[i] - best_centres[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random kick
                diff = np.random.randn(2)
                dist = np.linalg.norm(diff)
            
            dir_vec = diff / dist
            
            improved = False
            for _ in range(20):
                # Try moving i in dir_vec and j in -dir_vec
                # Add some noise
                angle_noise = np.random.uniform(-0.5, 0.5)
                cos_a = np.cos(angle_noise)
                sin_a = np.sin(angle_noise)
                
                rot_dir = np.array([dir_vec[0]*cos_a - dir_vec[1]*sin_a,
                                    dir_vec[0]*sin_a + dir_vec[1]*cos_a])
                
                new_c = best_centres.copy()
                new_c[i] += step_size * rot_dir
                new_c[j] -= step_size * rot_dir
                
                if np.any(new_c < -0.5) or np.any(new_c > 0.5):
                    continue
                    
                new_r = _compute_radius(new_c)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_c
                    improved = True
                    break
            
            if not improved:
                step_size *= 0.95
                if step_size < min_step:
                    break

    return best_centres, best_r

def construct():
    start_time = time.time()
    
    # 1. Initialization
    # Hexagonal packing is a strong start
    centres = _hexagonal_packing()
    current_r = _compute_radius(centres)
    
    # 2. Global Optimization with L-BFGS-B
    # Allocate most of the time to this
    time_elapsed = time.time() - start_time
    time_budget_opt = min(TIME_LIMIT - 20, 120.0)
    if time_elapsed < time_budget_opt:
        centres = _optimize_l_bfgs_b(centres, time_budget_opt)
        current_r = _compute_radius(centres)
    
    # 3. Local Polish
    time_elapsed = time.time() - start_time
    time_budget_polish = min(TIME_LIMIT - 5, 10.0)
    if time_elapsed < time_budget_polish:
        centres, current_r = _polish(centres, time_budget_polish)
    
    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
