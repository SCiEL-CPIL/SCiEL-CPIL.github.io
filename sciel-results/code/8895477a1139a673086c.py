"""
Optimization of 50 equal circles in a unit square.
Strategy:
1. Initialize with a hexagonal grid.
2. Use binary search on the radius 'r' to find the maximum feasible packing.
   - Feasibility is checked by minimizing constraint violations using projected gradient descent.
3. Final polish using L-BFGS on a soft-min objective to maximize precision.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _initial_grid():
    """Create initial hexagonal grid layout with slight jitter."""
    # Hexagonal packing: offset every other row
    # For N=50, a 7x8 grid or similar works well.
    # Let's try to fit 50 points in a hex grid.
    # Rows: 1, 2, ... 
    # Standard hex grid for square:
    # Spacing dx, dy = dx * sqrt(3)/2
    
    # We want to maximize the minimum distance.
    # A 7x8 grid has 56 spots. We take 50.
    rows = 7
    cols = 8
    
    # Calculate spacing to fit in [-0.5, 0.5]
    # Width: (cols - 1) * dx + dx = cols * dx <= 1.0 => dx <= 1/cols
    # But we need margin for radius.
    # Let's just place them centered.
    
    dx = 1.0 / cols
    dy = dx * np.sqrt(3) / 2
    
    # Total height: (rows - 1) * dy
    # Center vertically
    start_y = -0.5 + dy / 2 + (1.0 - (rows - 1) * dy) / 2
    start_x = -0.5 + dx / 2
    
    points = []
    for row in range(rows):
        y = start_y + row * dy
        offset = dx / 2 if row % 2 == 1 else 0
        for col in range(cols):
            x = start_x + col * dx + offset
            # Check if inside [-0.5, 0.5] with some margin
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            if len(points) >= N:
                break
        if len(points) >= N:
            break
    
    # If we don't have enough, fill in with square grid or random
    if len(points) < N:
        # Fallback: Square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for y in ticks:
            for x in ticks:
                if len(points) >= N:
                    break
                if not any(np.allclose([x, y], p) for p in points):
                    points.append([x, y])
            if len(points) >= N:
                break
        
        # If still not enough (should not happen for N=50 with 7x8 hex), add random
        while len(points) < N:
            p = [np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4)]
            points.append(p)
    
    grid = np.array(points[:N])
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.002, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _violation_objective(centers, r):
    """
    Calculate total violation of constraints for a given radius r.
    Violation = sum(max(0, overlap)) for all pairs + sum(max(0, wall_penetration)).
    This is a smooth (mostly) objective suitable for gradient descent.
    """
    centers = np.asarray(centers, dtype=np.float64)
    
    # Wall violations
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    wall_viol = np.sum(np.maximum(0, r - wall_min))
    
    # Pairwise violations
    # Use broadcasting for simplicity, N=50 is small enough (50x50 matrix)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Overlap if dist < 2*r
    overlaps = np.maximum(0, 2*r - dists)
    pair_viol = np.sum(overlaps)
    
    return wall_viol + pair_viol

def _violation_grad(centers, r):
    """
    Gradient of the violation objective.
    """
    centers = np.asarray(centers, dtype=np.float64)
    grad = np.zeros_like(centers)
    
    # Wall gradients
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    wall_min = np.minimum(wall_x, wall_y)
    
    for i in range(N):
        if r > wall_min[i]:
            if wall_x[i] <= wall_y[i]:
                # Pushing towards center in x
                grad[i, 0] -= np.sign(centers[i, 0])
            else:
                # Pushing towards center in y
                grad[i, 1] -= np.sign(centers[i, 1])
    
    # Pairwise gradients
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e-10) # Avoid div by zero
    
    # Gradient of max(0, 2r - d) is -1 * (d_vec / d) if 2r > d, else 0
    # d_vec = c_i - c_j
    # grad_i += (c_i - c_j) / d
    # grad_j -= (c_i - c_j) / d
    
    # We only care about pairs where dist < 2r
    mask = dists < 2*r
    # Set diagonal to false
    mask[np.diag_indices_from(mask)] = False
    
    # Vectorized update
    # For each pair (i, j) in mask:
    # grad[i] += diff[i,j] / dists[i,j]
    # grad[j] -= diff[i,j] / dists[i,j]
    
    # Using np.einsum or broadcasting
    # factor = mask / dists
    factor = np.zeros_like(dists)
    factor[mask] = 1.0 / dists[mask]
    
    # grad = sum_j factor[i,j] * diff[i,j]
    # diff[i,j] = c_i - c_j
    # sum_j factor[i,j] * (c_i - c_j) = c_i * sum_j factor[i,j] - sum_j factor[i,j] * c_j
    
    sum_factor = np.sum(factor, axis=1)
    weighted_sum = np.sum(factor[:, :, np.newaxis] * centers[np.newaxis, :, :], axis=1)
    
    grad += centers * sum_factor[:, np.newaxis] - weighted_sum
    
    return grad

def _feasibility_check(centers, r, time_budget):
    """
    Try to find a valid packing for radius r starting from 'centers'.
    Returns (success, best_centers, min_violation).
    Uses projected gradient descent.
    """
    start_time = time.time()
    current = centers.copy()
    best_centers = current.copy()
    best_viol = _violation_objective(current, r)
    
    lr = 0.05
    decay = 0.999
    max_iter = 10000
    
    for i in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        viol = _violation_objective(current, r)
        if viol < best_viol:
            best_viol = viol
            best_centers = current.copy()
            
        if viol < 1e-10:
            return True, best_centers, best_viol
        
        grad = _violation_grad(current, r)
        
        # Gradient descent step
        current -= lr * grad
        
        # Project back to square [-0.5, 0.5]
        current = np.clip(current, -0.5, 0.5)
        
        # Decay learning rate
        lr *= decay
        
        # Occasionally restart with small noise if stuck? 
        # Or just let it run.
        
    return False, best_centers, best_viol

def _lbfgs_polish(centers, time_budget):
    """
    Final polish using L-BFGS on soft-min objective to maximize radius.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        
        c = x.reshape(-1, 2)
        
        # Wall distances
        wall_dists_x = 0.5 - np.abs(c[:, 0])
        wall_dists_y = 0.5 - np.abs(c[:, 1])
        walls = np.minimum(wall_dists_x, wall_dists_y)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Upper triangle pairwise distances
        iu, ju = np.triu_indices(N, k=1)
        pair_upper = pair_dists[iu, ju]
        
        all_dists = np.concatenate([walls, pair_upper])
        
        # Soft-min
        tau = 0.001
        min_d = np.min(all_dists)
        shifted = all_dists - min_d
        log_sum = np.log(np.sum(np.exp(-shifted / tau)))
        soft_min = min_d - tau * log_sum
        
        obj = -soft_min
        
        # Gradient
        exp_vals = np.exp(-shifted / tau)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Wall gradient
        for i in range(N):
            w_wall = weights[i]
            if walls[i] == wall_dists_x[i]:
                grad[2*i] += w_wall * np.sign(c[i, 0])
            if walls[i] == wall_dists_y[i]:
                grad[2*i+1] += w_wall * np.sign(c[i, 1])
        
        # Pairwise gradient
        for k in range(len(iu)):
            w_pair = weights[N + k]
            i, j = iu[k], ju[k]
            d = dists[i, j]
            if d > 1e-12:
                factor = 0.5 * w_pair / d
                diff_ij = c[i] - c[j]
                grad[2*i:2*i+2] -= factor * diff_ij
                grad[2*j:2*j+2] += factor * diff_ij
        
        return obj, grad
    
    try:
        result = opt.minimize(
            obj_and_grad, current.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14,
                     'maxfun': 5000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        
        r_polished = _compute_radius(polished)
        if r_polished > best_r:
            best_r = r_polished
            best = polished
    except Exception:
        pass
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Phase 1: Initial Grid
    centers = _initial_grid()
    best_radius = _compute_radius(centers)
    
    # Phase 2: Binary Search on Radius
    # Lower bound: current radius
    # Upper bound: theoretical max for 50 circles? 
    # Area of 50 circles = 50 * pi * r^2 <= 1 => r <= 1/sqrt(50*pi) ~ 0.0797
    # Let's use 0.075 as a safe upper bound for search
    lo = best_radius
    hi = 0.075
    
    # If initial radius is already high, adjust hi
    if best_radius > 0.07:
         hi = best_radius * 1.05
    
    # Binary search iterations
    # We need to be careful with time.
    # Each feasibility check takes some time.
    
    # First, try to improve the initial configuration with a quick gradient polish
    # to get a better starting point for binary search.
    if time.time() < deadline - 30:
        # Quick L-BFGS polish
        polished, r_polished = _lbfgs_polish(centers, time_budget=min(10, deadline - time.time() - 30))
        if r_polished > best_radius:
            centers = polished
            best_radius = r_polished
            lo = best_radius
    
    # Binary Search
    # Number of iterations depends on time budget.
    # Let's allocate ~60% of remaining time to binary search.
    bs_budget = 0.6 * (deadline - time.time())
    
    # Estimate time per feasibility check.
    # We'll just run a loop and break if time is up.
    
    current_lo = lo
    current_hi = hi
    
    # To speed up feasibility check, we can use the previous best centers as start.
    # We maintain a 'best_feasible' configuration for the current 'lo'.
    best_feasible_centers = centers.copy()
    
    for _ in range(20): # Max 20 iterations
        if time.time() > deadline - 20: # Leave 20s for final polish
            break
            
        mid = (current_lo + current_hi) / 2
        
        # Check feasibility for 'mid'
        # Start from best_feasible_centers
        success, trial_centers, viol = _feasibility_check(
            best_feasible_centers, mid, 
            time_budget=min(5.0, (deadline - time.time()) * 0.4)
        )
        
        if success:
            current_lo = mid
            best_feasible_centers = trial_centers
            # Update best radius if this is better
            r_trial = _compute_radius(trial_centers)
            if r_trial > best_radius:
                best_radius = r_trial
                centers = trial_centers
        else:
            current_hi = mid
    
    # Phase 3: Final High-Precision Polish
    # Use the best feasible centers found
    if time.time() < deadline - 5:
        final_centers, final_radius = _lbfgs_polish(
            centers, 
            time_budget=min(15, deadline - time.time() - 5)
        )
        
        if final_radius > best_radius:
            best_radius = final_radius
            centers = final_centers
    
    # Validate
    if not np.all(np.isfinite(centers)) or centers.shape != (N, 2):
        return _initial_grid()
        
    return centers

if __name__ == "__main__":
    # Simple test
    c = construct()
    r = _compute_radius(c)
    print(f"Radius: {r}")
