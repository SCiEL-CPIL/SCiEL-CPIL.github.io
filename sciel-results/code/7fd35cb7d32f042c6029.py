"""
Optimized packing of 50 equal circles in a unit square.
Combines vectorized Softplus violation objective with L-BFGS-B optimization 
and a binary search strategy for the maximum radius.
"""
import numpy as np
import time
import scipy.optimize as opt
import sys

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """
    Compute the true maximum radius for a given configuration.
    Returns 0.0 if invalid.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5) or \
       np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
        return 0.0

    # Wall distances: min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Equivalent to 0.5 - |x|, 0.5 - |y|
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    # Vectorized calculation of all pairwise distances
    # diff[i, j, k] = centers[i, k] - centers[j, k]
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    Uses Softplus to smooth the constraints.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    # Pairwise distances
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    # Softplus: log(1 + exp(z)) / beta
    # Clip to avoid overflow
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)
    
    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta
    
    return float(np.sum(soft_wall) + np.sum(soft_pair))

def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradients
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
    # Vectorized wall gradient contribution
    # d/dx_i (softplus(beta*(r - (0.5 - |x_i|))))
    # = sig(beta*(r - wall)) * beta * d/dx_i (r - wall)
    # wall = min(0.5-|x|, 0.5-|y|)
    # If wall = 0.5-|x|, d(wall)/dx = -sign(x), so d(viol)/dx = -sign(x)
    # If wall = 0.5-|y|, d(wall)/dy = -sign(y), so d(viol)/dy = -sign(y)
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    # Note: The derivative of softplus(z)/beta is sigmoid(z).
    # The chain rule factor is beta * (derivative of violation).
    # violation = r - wall.
    # d(violation)/d(coord) = - d(wall)/d(coord).
    # d(0.5 - |x|)/dx = -sign(x).
    # So d(violation)/dx = sign(x).
    
    # Wait, let's re-verify signs.
    # v = r - (0.5 - |x|) = r - 0.5 + |x|
    # d(v)/dx = sign(x)
    # So grad += sig * sign(x)
    
    idx_wx = np.nonzero(wx)[0]
    if len(idx_wx) > 0:
        grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
        
    idx_wy = np.nonzero(wy)[0]
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
        
    idx_wt = np.nonzero(wt)[0]
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj  # shape (_NPAIRS, 2)
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    # d(pair_viol)/d(ci) = - d(dist)/d(ci) = - (ci - cj) / dist
    # d(pair_viol)/d(cj) = + d(dist)/d(cj) = + (ci - cj) / dist
    # Grad term = sig_pair * d(viol)/d(coord)
    
    # factor_k = sig_pair[k] / dists[k]
    factor = sig_pair / dists_safe
    # contribution per pair: factor * diff_vec
    contrib = factor[:, None] * diff_vec
    
    # Scatter
    # grad_i -= contrib
    # grad_j += contrib
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _hex_init(rng):
    """Create a hexagonal packing initial layout."""
    # Try a few different hexagonal orientations/densities
    best_grid = None
    best_r = 0
    
    for attempt in range(5):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        # Estimate spacing to fit in square
        # Width: cols * 2r <= 1 => r <= 1/(2*cols)
        # Height: rows * r*sqrt(3) <= 1 => r <= 1/(rows*sqrt(3))
        # But with offsets, it's slightly different. Just try a scale.
        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)
        
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        if len(centers) < N:
            # Pad with random points if we didn't generate enough (shouldn't happen with cols=ceil(N/rows))
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()
            
    return best_grid

def _grid_init(rng):
    """Create a grid layout with jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    grid = grid + rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    return grid

def _optimize_for_radius(centers, r_target, time_budget, beta=10.0):
    """
    Run L-BFGS-B to minimize violation for a fixed r_target.
    Returns (optimized_centers, violation).
    """
    start_time = time.time()
    x0 = centers.flatten()
    
    def objective_and_grad(x):
        v = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return v, g
        
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = opt.minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 1000, 'disp': False}
        )
        if time.time() - start_time > time_budget:
            # Return current best if timeout, but we don't track it in this simple wrapper
            # For now, just return result if available, else x0
            if result.x is not None:
                new_centers = result.x.reshape(N, 2)
            else:
                new_centers = centers
        else:
            new_centers = result.x.reshape(N, 2)
            
        final_viol = _soft_violation_obj(new_centers.flatten(), r_target, beta)
        return new_centers, final_viol
    except Exception as e:
        # If optimization fails, return original
        return centers, _soft_violation_obj(centers.flatten(), r_target, beta)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # 1. Generate multiple initial configurations
    inits = []
    # Hexagonal
    inits.append(_hex_init(rng))
    # Grid
    inits.append(_grid_init(rng))
    # Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Evaluate initial radii to find a good starting point
    best_init = None
    best_init_r = 0
    for init in inits:
        r = _compute_radius(init)
        if r > best_init_r:
            best_init_r = r
            best_init = init
            
    # 2. Binary Search for Maximum Radius
    # Known best is ~0.071377
    # Start with a range that is likely to contain the optimum
    r_low = best_init_r
    r_high = 0.075  # Slightly above known best
    
    best_pts = best_init
    best_radius = best_init_r
    
    # We will perform a binary search.
    # To save time, we limit the number of iterations and the time per iteration.
    max_iterations = 12
    time_per_iter = (DEADLINE - 20) / max_iterations  # Reserve 20s for final polish
    
    for i in range(max_iterations):
        if time.time() > deadline - 25:
            break
            
        r_mid = (r_low + r_high) / 2
        
        # Use the best known feasible configuration as the starting point for optimization
        # If r_mid is smaller than current best, we can likely just use best_pts
        # If r_mid is larger, we need to optimize.
        
        # Optimization budget for this iteration
        budget = time_per_iter - (time.time() - start_time - i*time_per_iter) # Rough estimate
        budget = max(1.0, min(10.0, time_per_iter * 0.8))
        
        # If r_mid is significantly larger than best_radius, start from best_pts
        # If r_mid is smaller, we might just check if best_pts works (it should, by definition of r_low)
        # Actually, if r_mid > best_radius, we must optimize.
        
        candidate_pts = best_pts.copy()
        
        # Run optimization
        optimized_pts, viol = _optimize_for_radius(candidate_pts, r_mid, budget, beta=10.0)
        
        # Check feasibility
        actual_r = _compute_radius(optimized_pts)
        
        if actual_r >= r_mid - 1e-6:
            # Feasible
            r_low = r_mid
            if actual_r > best_radius:
                best_radius = actual_r
                best_pts = optimized_pts.copy()
        else:
            # Not feasible
            r_high = r_mid
            # We can try to use the optimized points as a better start for next time?
            # Only if they are closer to feasibility (lower violation)
            # But for simplicity, keep best_pts as the last known feasible
    
    # 3. Final Polish
    # Use the best radius found and polish it with a higher beta for smoother convergence
    if time.time() < deadline - 5:
        # Increase beta for finer resolution
        beta_final = 50.0
        # Run a few iterations of L-BFGS with higher beta
        current_pts = best_pts
        current_r = best_radius
        
        # Try to push radius slightly higher
        r_target = current_r * 1.001
        optimized_pts, viol = _optimize_for_radius(current_pts, r_target, 10.0, beta=beta_final)
        new_r = _compute_radius(optimized_pts)
        if new_r > current_r:
            best_radius = new_r
            best_pts = optimized_pts
        
        # Final strict polish at the found radius
        optimized_pts, viol = _optimize_for_radius(best_pts, best_radius, 5.0, beta=100.0)
        new_r = _compute_radius(optimized_pts)
        if new_r > best_radius:
            best_radius = new_r
            best_pts = optimized_pts
            
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# For testing
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
    print(f"Shape: {pts.shape}")
