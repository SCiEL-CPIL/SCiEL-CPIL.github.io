"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys


def _compute_score(centers):
    """Compute the minimum radius for the given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Vectorized pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _project_feasible(centers, r, max_iter=200, tol=1e-10):
    """
    Project centers onto the feasible set for radius r using iterative
    separation. Returns (feasible_centers, is_feasible).
    """
    n = len(centers)
    c = centers.copy()
    
    for iteration in range(max_iter):
        # Compute pairwise distances and wall distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        min_pair = np.min(dists) / 2.0
        
        # Check feasibility
        if min_pair >= r - tol and min_wall >= r - tol:
            return c, True
        
        # Compute forces: push apart overlapping pairs, push away from walls
        force = np.zeros_like(c)
        
        # Pairwise separation forces
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < 2 * r:
                    # Need to push apart by (2r - d)
                    if d > 1e-12:
                        push = (2 * r - d) / d
                        direction = (c[i] - c[j]) / d
                        force[i] += push * direction / 2
                        force[j] -= push * direction / 2
                    else:
                        # Coincident: random small push
                        rand_dir = np.random.randn(3)
                        rand_dir /= np.linalg.norm(rand_dir)
                        force[i] += rand_dir * 1e-3
                        force[j] -= rand_dir * 1e-3
        
        # Wall forces
        for i in range(n):
            for k in range(3):
                if c[i, k] > 0:
                    dist_to_wall = 0.5 - c[i, k]
                    if dist_to_wall < r:
                        push = (r - dist_to_wall) / max(r, 1e-9)
                        force[i, k] -= push
                else:
                    dist_to_wall = c[i, k] + 0.5
                    if dist_to_wall < r:
                        push = (r - dist_to_wall) / max(r, 1e-9)
                        force[i, k] += push
        
        # Apply forces with adaptive step size
        step = 0.5
        c_new = c + step * force
        c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check if we improved
        new_min_pair = np.min(np.sqrt(np.sum((c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]) ** 2, axis=2))) / 2.0
        new_min_wall = np.min(0.5 - np.abs(c_new))
        new_score = min(new_min_pair, new_min_wall)
        
        old_score = min(min_pair, min_wall)
        
        if new_score >= old_score - 1e-12:
            c = c_new
        else:
            # Try smaller step
            step /= 2
            c_new = c + step * force
            c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
            c = c_new
        
        # Early termination if converged
        if np.max(np.abs(c - centers)) < tol:
            break
    
    # Final feasibility check
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    min_wall = np.min(0.5 - np.abs(c))
    
    return c, (min_pair >= r - tol and min_wall >= r - tol)


def _solve_for_radius(r, initial_centers, time_budget, max_iters=500):
    """
    Try to find a feasible configuration for radius r.
    Uses iterative separation starting from initial_centers.
    Returns (centers, success).
    """
    start_time = time.time()
    c = initial_centers.copy()
    
    # Multiple restarts with different initial perturbations
    for attempt in range(3):
        if time.time() - start_time > time_budget:
            break
        
        if attempt == 0:
            c_try = initial_centers.copy()
        else:
            c_try = initial_centers + np.random.normal(0, 0.01 * attempt, size=(N, 3))
            c_try = np.clip(c_try, -0.49, 0.49)
        
        feasible_c, is_feasible = _project_feasible(c_try, r, max_iter=max_iters)
        
        if is_feasible:
            # Verify
            diff = feasible_c[:, np.newaxis, :] - feasible_c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            min_wall = np.min(0.5 - np.abs(feasible_c))
            if min(min_pair, min_wall) >= r - 1e-9:
                return feasible_c, True
    
    return c, False


def _bisection_search(initial_centers, time_budget, r_low=None, r_high=None):
    """
    Bisection search for the maximum radius.
    """
    start_time = time.time()
    
    if r_low is None:
        r_low = _compute_score(initial_centers)
    if r_high is None:
        # Conservative upper bound
        r_high = 0.2
    
    # First, try to find a feasible point at r_high
    feasible_c, success = _solve_for_radius(r_high, initial_centers, 
                                             time_budget=min(10.0, time_budget / 4))
    if success:
        r_low = r_high
        # Try higher
        r_high = r_high * 1.1
        feasible_c, success = _solve_for_radius(r_high, initial_centers,
                                                 time_budget=min(10.0, time_budget / 4))
        if not success:
            r_high = r_high * 1.05
    
    # Bisection
    best_centers = initial_centers.copy()
    best_r = r_low
    
    for iteration in range(30):
        if time.time() - start_time > time_budget - 5.0:
            break
        
        r_mid = (r_low + r_high) / 2
        remaining = time_budget - (time.time() - start_time)
        
        if remaining < 5.0:
            break
        
        time_per_iter = min(remaining / 4, 15.0)
        
        feasible_c, success = _solve_for_radius(r_mid, initial_centers, 
                                                 time_budget=time_per_iter)
        
        if success:
            r_low = r_mid
            if r_mid > best_r:
                best_r = r_mid
                best_centers = feasible_c.copy()
        else:
            r_high = r_mid
        
        # Convergence check
        if r_high - r_low < 1e-8:
            break
    
    return best_centers, best_r


def _soft_min_radius(centers, beta=500.0):
    """
    Compute a soft-min approximation of the minimum radius.
    Uses log-sum-exp for the pairwise distances and wall distances.
    """
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Wall distances squared
    wall_dists_sq = (0.5 - np.abs(centers)) ** 2
    wall_dists_sq_flat = wall_dists_sq.flatten()
    
    pair_vals = dists_sq / 4.0
    np.fill_diagonal(pair_vals, np.inf)
    pair_vals_flat = pair_vals.flatten()
    
    all_vals = np.concatenate([pair_vals_flat, wall_dists_sq_flat])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return np.sqrt(soft_min)


def _optimize_smooth(centers, time_budget, beta=200.0):
    """
    Optimize using smooth surrogate objective with scipy L-BFGS-B.
    """
    start_time = time.time()
    
    x0 = centers.flatten()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return 1e10
        c = x.reshape(-1, 3)
        return -_soft_min_radius(c, beta)
    
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    best_centers = result.x.reshape(-1, 3)
    best_score = _compute_score(best_centers)
    
    return best_centers, best_score


def _local_search(centers, time_budget):
    """
    Local search using coordinate descent and small perturbations.
    """
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    while time.time() - start_time < time_budget:
        improved = False
        
        for idx in range(N):
            if time.time() - start_time > time_budget:
                break
            
            for dim in range(3):
                for step in step_sizes:
                    for direction in [-1, 1]:
                        new_centers = current.copy()
                        new_centers[idx, dim] += direction * step
                        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                        
                        new_score = _compute_score(new_centers)
                        if new_score > current_score + 1e-12:
                            current = new_centers
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best = current.copy()
                            break
        
        if not improved:
            # Try random perturbations
            for _ in range(50):
                if time.time() - start_time > time_budget:
                    break
                
                idx = np.random.randint(N)
                delta = np.random.randn(3) * 0.001
                new_centers = current.copy()
                new_centers[idx] += delta
                new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                
                new_score = _compute_score(new_centers)
                if new_score > current_score + 1e-12:
                    current = new_centers
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best = current.copy()
        
        if not improved:
            break
    
    return best, best_score


def _bcc_initial():
    """Generate a BCC-like initial configuration for 50 spheres."""
    m = 4
    spacing = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
    
    centers = np.array(centers[:N])
    return centers


def _hcp_initial():
    """
    Generate an HCP-like initial configuration.
    """
    centers = []
    a = 0.25
    n_layers = 5
    layer_spacing = a * np.sqrt(2/3)
    
    for layer in range(n_layers):
        z = -0.5 + 0.15 + layer * layer_spacing
        if z > 0.45:
            break
        
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * a / 2
                y = j * a * np.sqrt(3) / 2
                
                if layer % 2 == 1:
                    x += a / 2
                    y += a * np.sqrt(3) / 6
                
                if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                    centers.append([x, y, z])
    
    centers = np.array(centers)
    if len(centers) > N:
        centers = centers[:N]
    elif len(centers) < N:
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.4, 0.4, 3)])
    
    return centers


def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = [
        _bcc_initial(),
        _hcp_initial(),
        _bcc_initial() + np.random.normal(0, 0.02, size=(N, 3)),
    ]
    
    for i in range(len(initial_configs)):
        initial_configs[i] = np.clip(initial_configs[i], -0.49, 0.49)
    
    best_centers = initial_configs[0].copy()
    best_score = _compute_score(best_centers)
    
    # Strategy 1: Bisection search with projected feasibility
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 30.0:
        time_for_bisection = min(remaining / 3, 40.0)
        
        for i, initial in enumerate(initial_configs):
            elapsed = time.time() - start_time
            remaining = deadline - elapsed
            if remaining < 10.0:
                break
            
            time_alloc = min(remaining / (len(initial_configs) - i), time_for_bisection)
            
            bisected, bisected_r = _bisection_search(initial, time_budget=time_alloc)
            
            if bisected_r > best_score:
                best_score = bisected_r
                best_centers = bisected.copy()
    
    # Strategy 2: Smooth optimization to refine
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 20.0:
        time_for_smooth = min(remaining / 2, 25.0)
        optimized, score = _optimize_smooth(best_centers, time_budget=time_for_smooth)
        
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
    
    # Strategy 3: Local search polish
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 10.0:
        polished, polished_score = _local_search(best_centers, time_budget=min(remaining - 3.0, 25.0))
        if polished_score > best_score:
            best_score = polished_score
            best_centers = polished.copy()
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5.0:
        polished2, polished_score2 = _local_search(best_centers, time_budget=min(remaining - 2.0, 12.0))
        if polished_score2 > best_score:
            best_score = polished_score2
            best_centers = polished2.copy()
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
