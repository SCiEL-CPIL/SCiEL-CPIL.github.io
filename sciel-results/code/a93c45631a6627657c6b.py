"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    # Wall distances
    wall_dists = np.minimum(0.5 - np.abs(centers), 0.5)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    n = len(centers)
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _hexagonal_initial(n):
    """Generate a hexagonal packing initial configuration."""
    # For hexagonal packing in a unit square [-0.5, 0.5]^2
    # The optimal hexagonal spacing for density
    # Start with a grid and perturb
    
    # Number of points per row for hexagonal arrangement
    # Hexagonal: rows offset by half spacing
    # Let's use a base spacing
    # For 200 points, try m x k grid with hexagonal offset
    
    # Try different arrangements
    best_centers = None
    best_radius = -1
    
    for m in range(10, 20):
        k = int(np.ceil(n / m))
        if m * k < n:
            continue
            
        # Hexagonal spacing
        # Horizontal spacing
        sx = 1.0 / m
        # Vertical spacing for hexagonal
        sy = sx * np.sqrt(3) / 2
        
        # Check if it fits
        total_height = (k - 1) * sy
        if total_height > 1.0:
            continue
            
        centers = np.zeros((n, 2))
        count = 0
        for j in range(k):
            for i in range(m):
                if count >= n:
                    break
                x = (i + 0.5) * sx - 0.5
                # Offset odd rows
                if j % 2 == 1:
                    x += sx / 2
                y = (j + 0.5) * sy - total_height / 2
                
                # Clamp to stay in bounds
                x = np.clip(x, -0.5, 0.5)
                y = np.clip(y, -0.5, 0.5)
                
                centers[count] = [x, y]
                count += 1
            if count >= n:
                break
        
        if count < n:
            continue
            
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers

def _energy(centers, t):
    """
    Energy function for optimization.
    We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    
    Use a smooth approximation:
    E = -r where r is approximated smoothly
    
    Or use: E = sum of violations + penalty for small distances
    
    Better: minimize -log(sum(exp(-d_i/t)) for all constraints)
    where d_i are the "slack" values (distances - 2r for pairs, wall_dist - r for walls)
    
    Actually, let's use a direct approach:
    Maximize r subject to:
    - |c_i - c_j| >= 2r for all i < j
    - |c_ik| <= 0.5 - r for all i, k
    
    This is a non-convex problem. Let's use a penalty method or direct optimization.
    
    For the smooth surrogate:
    Define for each constraint a "violation" v_i = max(0, required - actual)
    E = sum(v_i^2) + lambda * (-r)
    
    But r is also a variable. Let's fix r and check feasibility, or optimize r.
    
    Alternative: directly optimize to maximize the minimum distance.
    Use: f(centers) = min over all constraints of the "slack"
    
    Let's use the approach of minimizing a smooth approximation of the negative of the minimum.
    """
    n = len(centers)
    r = t  # Use t as the target radius
    
    # Wall constraints: need 0.5 - |c_ik| >= r
    # Slack for walls: 0.5 - |c_ik| - r
    wall_slacks = 0.5 - np.abs(centers) - r
    min_wall_slack = np.min(wall_slacks)
    
    # Pair constraints: need |c_i - c_j| >= 2r
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    pair_slack = min_pair_dist - 2 * r
    
    # The actual radius is min(0.5 - max|c_ik|, min_pair_dist/2)
    actual_r = min(-min_wall_slack + r, pair_slack / 2 + r)
    # Wait, that's circular. Let me redo.
    
    # actual_r = min(0.5 - max(|c_ik|), min_pair_dist / 2)
    max_abs_coord = np.max(np.abs(centers))
    actual_r = min(0.5 - max_abs_coord, min_pair_dist / 2)
    
    return -actual_r  # We want to minimize this (maximize actual_r)

def _gradient_energy(centers, t):
    """Gradient of the energy function."""
    n = len(centers)
    max_abs_coord = np.max(np.abs(centers))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    actual_r = min(0.5 - max_abs_coord, min_pair_dist / 2)
    
    grad = np.zeros_like(centers)
    
    # If wall constraint is active
    if 0.5 - max_abs_coord <= min_pair_dist / 2:
        # Gradient from wall constraint
        # d(0.5 - max|c_ik|)/dc_ik = -sign(c_ik) for the max element
        for i in range(n):
            for k in range(2):
                if abs(centers[i, k]) == max_abs_coord:
                    grad[i, k] = -np.sign(centers[i, k])
    else:
        # Gradient from pair constraint
        # Find the pair with minimum distance
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        if i == j:
            return grad
        diff_ij = centers[i] - centers[j]
        dist_ij = dists[i, j]
        if dist_ij > 1e-12:
            # d(min_dist/2)/dc_i = diff_ij / (2 * dist_ij)
            # d(min_dist/2)/dc_j = -diff_ij / (2 * dist_ij)
            grad[i] += diff_ij / (2 * dist_ij)
            grad[j] -= diff_ij / (2 * dist_ij)
    
    return grad

def construct():
    """Construct the optimal packing configuration."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Initial configuration
    centers = _hexagonal_initial(N)
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Local optimization using BFGS
    # We'll do multiple restarts with different initial conditions
    
    n_restarts = 5
    attempts = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            x0 = best_centers.copy()
        else:
            # Add random perturbation
            x0 = best_centers + np.random.normal(0, 0.01, size=best_centers.shape)
            # Ensure within bounds
            x0 = np.clip(x0, -0.49, 0.49)
        
        # Use BFGS to maximize radius (minimize negative radius)
        def objective(x):
            return -_compute_radius(x.reshape(N, 2))
        
        def gradient(x):
            c = x.reshape(N, 2)
            return _gradient_energy(c, 0).reshape(-1)
        
        try:
            result = minimize(
                objective,
                x0.reshape(-1),
                method='BFGS',
                jac=gradient,
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            
            x_opt = result.x.reshape(N, 2)
            # Ensure within bounds
            x_opt = np.clip(x_opt, -0.5, 0.5)
            
            r_opt = _compute_radius(x_opt)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = x_opt.copy()
        except Exception:
            pass
        
        attempts += 1
        if time.time() > deadline - 15:
            break
    
    # Final polish: use a more precise optimization
    if time.time() < deadline - 10:
        # Try a few more iterations with tighter tolerance
        x0 = best_centers.reshape(-1)
        
        def objective(x):
            return -_compute_radius(x.reshape(N, 2))
        
        def gradient(x):
            c = x.reshape(N, 2)
            return _gradient_energy(c, 0).reshape(-1)
        
        try:
            result = minimize(
                objective,
                x0,
                method='BFGS',
                jac=gradient,
                options={'maxiter': 1000, 'gtol': 1e-10}
            )
            
            x_opt = result.x.reshape(N, 2)
            x_opt = np.clip(x_opt, -0.5, 0.5)
            
            r_opt = _compute_radius(x_opt)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = x_opt.copy()
        except Exception:
            pass
    
    return best_centers
# EVOLVE-BLOCK-END
