"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum feasible radius for given centers."""
    # Wall constraint
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall)
    
    # Pairwise constraint
    if len(centers) > 1:
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dist, np.inf)
        r_pair = np.min(dist) / 2.0
    else:
        r_pair = np.inf
    
    return min(r_wall, r_pair)


def _score(centers):
    """Score function: maximize the minimum radius."""
    return _compute_radius(centers)


def _energy_and_grad(centers, r_target, beta=100.0):
    """
    Energy function: penalize violations of the constraint that spheres of radius r_target
    must be non-overlapping and inside the cube.
    
    We use a soft penalty: for each pair (i,j), if dist(i,j) < 2*r_target, penalize.
    For walls, if |c_ik| > 0.5 - r_target, penalize.
    
    Returns (energy, gradient).
    """
    n = len(centers)
    centers = centers.reshape(n, 3)
    
    # Wall violations
    # For each point, for each dimension, violation = max(0, |c| - (0.5 - r_target))
    wall_margin = 0.5 - r_target
    abs_c = np.abs(centers)
    wall_viol = np.maximum(0, abs_c - wall_margin)
    
    # Wall energy and gradient
    wall_energy = np.sum(wall_viol ** 2)
    # Gradient: 2 * wall_viol * sign(c) for violated components
    wall_grad = np.zeros_like(centers)
    mask_wall = wall_viol > 0
    wall_grad[mask_wall] = 2 * wall_viol[mask_wall] * np.sign(centers[mask_wall])
    
    # Pairwise violations
    # For each pair, if dist < 2*r_target, violation = 2*r_target - dist
    if n > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
        np.fill_diagonal(dist, np.inf)
        
        min_dist = 2 * r_target
        pair_viol = np.maximum(0, min_dist - dist)
        
        pair_energy = np.sum(pair_viol ** 2)
        
        # Gradient computation
        # d(pair_viol)/d(centers_i) = -diff[i,j] / dist[i,j] * (1 if violated)
        pair_grad = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                if pair_viol[i, j] > 0:
                    d = diff[i, j]
                    dist_ij = dist[i, j]
                    grad_ij = 2 * pair_viol[i, j] * (-d / dist_ij)
                    pair_grad[i] += grad_ij
                    pair_grad[j] -= grad_ij
    
    else:
        pair_energy = 0
        pair_grad = np.zeros_like(centers)
    
    total_energy = wall_energy + pair_energy
    total_grad = wall_grad + pair_grad
    
    return total_energy, total_grad


def _try_radius(centers, r_target, max_iter=1000, tol=1e-8):
    """
    Try to find a configuration where all spheres of radius r_target fit.
    Uses gradient descent to minimize the violation energy.
    Returns (success, new_centers).
    """
    n = len(centers)
    x0 = centers.reshape(-1).copy()
    
    def objective(x):
        c = x.reshape(n, 3)
        e, g = _energy_and_grad(c, r_target)
        return e, g
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (3 * n)
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-5})
    
    new_centers = result.x.reshape(n, 3)
    final_energy = result.fun
    
    # Check if constraints are satisfied
    # Recompute the actual radius
    r_actual = _compute_radius(new_centers)
    
    success = r_actual >= r_target - 1e-10
    return success, new_centers


def _local_search(centers, r_target, n_restarts=3, max_iter_per_restart=500):
    """
    Try multiple restarts from slightly perturbed positions to find a valid configuration
    for the given radius.
    """
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = centers.copy()
        else:
            # Small random perturbation
            noise = np.random.normal(0, 0.005, size=centers.shape)
            x0 = centers + noise
            # Clamp to cube
            x0 = np.clip(x0, -0.5, 0.5)
        
        success, new_centers = _try_radius(x0, r_target, max_iter=max_iter_per_restart)
        r_new = _compute_radius(new_centers)
        
        if r_new > best_r:
            best_r = r_new
            best_centers = new_centers.copy()
        
        if success:
            return True, new_centers
    
    return False, best_centers


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin
    
    # Initial guess: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Compute initial radius
    r_current = _compute_radius(centers)
    
    # Binary search for the optimal radius
    # Lower bound: current radius
    # Upper bound: theoretical max (cube side / 2 / something)
    # For 100 spheres in unit cube, rough upper bound
    r_low = r_current
    r_high = 0.5 / (N ** (1.0/3.0)) * 1.5  # Loose upper bound
    
    # Actually, let's use a better upper bound
    # Volume argument: 100 * (4/3) pi r^3 <= 1
    # r <= (3/(400 pi))^(1/3) ≈ 0.191
    r_high = (3.0 / (4.0 * np.pi * N)) ** (1.0/3.0)
    
    # Also wall constraint: r <= 0.5
    r_high = min(r_high, 0.45)
    
    # Binary search
    best_centers = centers.copy()
    best_r = r_current
    
    # Do a few iterations of binary search
    n_binary_iters = 10
    for it in range(n_binary_iters):
        if time.time() > deadline - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        success, new_centers = _local_search(centers, r_mid, n_restarts=2, max_iter_per_restart=300)
        
        if success:
            r_low = r_mid
            centers = new_centers
            best_r = _compute_radius(centers)
            best_centers = centers.copy()
        else:
            r_high = r_mid
            # Use the best centers found during the attempt
            r_attempt = _compute_radius(new_centers)
            if r_attempt > best_r:
                best_r = r_attempt
                best_centers = new_centers.copy()
    
    # Final polish: try to improve from the best configuration
    if time.time() < deadline - 5:
        # Try a few more local improvements
        for polish_iter in range(3):
            if time.time() > deadline - 5:
                break
            
            # Try to increase radius slightly
            r_target = best_r * 1.01
            success, new_centers = _local_search(best_centers, r_target, n_restarts=2, max_iter_per_restart=200)
            
            if success:
                best_r = _compute_radius(new_centers)
                best_centers = new_centers.copy()
            else:
                r_attempt = _compute_radius(new_centers)
                if r_attempt > best_r:
                    best_r = r_attempt
                    best_centers = new_centers.copy()
                break
    
    return best_centers

# EVOLVE-BLOCK-END
