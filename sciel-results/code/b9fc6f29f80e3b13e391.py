"""
Optimized packing of 50 spheres in a unit cube.
Strategy:
1. Initialize with diverse configurations (grid, FCC-inspired, random).
2. Run a damped molecular dynamics simulation to escape local minima.
3. Polish the result using L-BFGS-B with a vectorized soft-min objective for high precision.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 50
CUBE_HALF = 0.5
EPS = 1e-12

def _compute_score(centers):
    """Compute the packing radius for a set of centers."""
    centers = np.clip(centers, -CUBE_HALF + EPS, CUBE_HALF - EPS)
    
    # Wall distance
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_obj_and_grad(params, beta, centers_shape=(N, 3)):
    """
    Objective and gradient: minimize -soft_min(distances)
    Returns (value, gradient)
    """
    centers = params.reshape(centers_shape)
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    # Precomputed indices for upper triangle
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_full = np.sqrt(np.sum(diff**2, axis=-1))
    d_pairs = d_full[tri_i, tri_j] / 2.0
    
    # Wall distances
    d_x_pos = CUBE_HALF - centers[:, 0]
    d_x_neg = centers[:, 0] + CUBE_HALF
    d_y_pos = CUBE_HALF - centers[:, 1]
    d_y_neg = centers[:, 1] + CUBE_HALF
    d_z_pos = CUBE_HALF - centers[:, 2]
    d_z_neg = centers[:, 2] + CUBE_HALF
    
    wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
    all_dists = np.concatenate([d_pairs, wall_dists])
    
    # Soft minimum calculation
    min_d = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - np.log(np.sum(exp_vals)) / beta
    
    # Weights for gradient
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient computation
    grad_centers = np.zeros_like(centers)
    
    # Pairwise gradient
    diff_ij = centers[tri_i] - centers[tri_j]
    d_ij = d_full[tri_i, tri_j]
    safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
    unit_vecs = diff_ij / safe_d[:, np.newaxis]
    w_pair = weights[:n_pairs] * 0.5 / safe_d
    contrib = w_pair[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_centers, tri_i, contrib)
    np.add.at(grad_centers, tri_j, -contrib)
    
    # Wall gradients
    ws = n_pairs
    grad_centers[:, 0] -= weights[ws:ws+N]
    grad_centers[:, 0] += weights[ws+N:ws+2*N]
    grad_centers[:, 1] -= weights[ws+2*N:ws+3*N]
    grad_centers[:, 1] += weights[ws+3*N:ws+4*N]
    grad_centers[:, 2] -= weights[ws+4*N:ws+5*N]
    grad_centers[:, 2] += weights[ws+5*N:ws+6*N]
    
    return -soft_min_val, -grad_centers.flatten()

def _dynamics_phase(centers, time_budget):
    """
    Run a damped molecular dynamics simulation with adaptive shaking to maximize the minimum distance.
    """
    positions = centers.copy().astype(np.float64)
    velocities = np.zeros_like(positions)
    
    k_force = 1.0
    damping = 0.95
    dt = 0.005
    shake_interval = 500  # shake every N steps
    step_count = 0
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(positions)
        
        # Pairwise forces - use stronger repulsion for close pairs
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Use inverse square law for stronger repulsion when close
        min_dist = 0.1
        inv_r2 = np.where(dists > min_dist, 1.0 / (dists**2), 1.0 / (min_dist**2))
        np.fill_diagonal(inv_r2, 0.0)
        
        force_mat = k_force * diff * inv_r2[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        
        # Wall forces - stronger when close to walls
        d_left = np.clip(positions[:, 0] + CUBE_HALF, 1e-6, None)
        d_right = np.clip(CUBE_HALF - positions[:, 0], 1e-6, None)
        d_bottom = np.clip(positions[:, 1] + CUBE_HALF, 1e-6, None)
        d_top = np.clip(CUBE_HALF - positions[:, 1], 1e-6, None)
        d_back = np.clip(positions[:, 2] + CUBE_HALF, 1e-6, None)
        d_front = np.clip(CUBE_HALF - positions[:, 2], 1e-6, None)
        
        wall_strength = 2.0
        forces[:, 0] += wall_strength * k_force / (d_left**2) - wall_strength * k_force / (d_right**2)
        forces[:, 1] += wall_strength * k_force / (d_bottom**2) - wall_strength * k_force / (d_top**2)
        forces[:, 2] += wall_strength * k_force / (d_back**2) - wall_strength * k_force / (d_front**2)
        
        # Adaptive damping: increase damping as we converge
        current_damping = damping * (1.0 + 0.001 * step_count)
        current_damping = min(current_damping, 0.99)
        
        velocities = velocities * current_damping + forces * dt
        positions = positions + velocities * dt
        positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        step_count += 1
        
        # Periodic shaking to escape local minima
        if step_count % shake_interval == 0:
            # Add random perturbation proportional to current velocity scale
            vel_scale = np.mean(np.abs(velocities))
            shake_strength = 0.01 * max(vel_scale, 0.001)
            velocities += shake_strength * np.random.randn(N, 3)
    
    return positions

def _build_initial_grid():
    """
    Build a good initial configuration using a combination of lattice structures and space-filling curves.
    """
    np.random.seed(42)
    
    # Strategy: Use multiple lattice structures and combine them
    # 1. Simple cubic grid (5x5x4 = 100 points, take 50)
    sc_points = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                x = (ix - 2) * 0.2
                y = (iy - 2) * 0.2
                z = (iz - 1.5) * 0.25
                if abs(x) < 0.48 and abs(y) < 0.48 and abs(z) < 0.48:
                    sc_points.append([x, y, z])
    
    # 2. BCC-inspired structure
    bcc_points = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                x = (ix - 1.5) * 0.28
                y = (iy - 1.5) * 0.28
                z = (iz - 1.5) * 0.28
                if abs(x) < 0.48 and abs(y) < 0.48 and abs(z) < 0.48:
                    bcc_points.append([x, y, z])
    
    # 3. Random points with minimum distance constraint
    random_points = []
    attempts = 0
    while len(random_points) < N and attempts < 10000:
        p = np.random.uniform(-0.45, 0.45, size=3)
        if len(random_points) == 0:
            random_points.append(p)
        else:
            dists = np.sqrt(np.sum((np.array(random_points) - p)**2, axis=1))
            if np.min(dists) > 0.15:
                random_points.append(p)
        attempts += 1
    
    # Combine and select best N points
    all_points = sc_points[:N] + bcc_points[:N] + random_points[:N]
    all_points = np.array(all_points)
    
    # Use greedy selection: pick points that maximize minimum distance to already selected
    selected = [all_points[0]]
    for _ in range(N - 1):
        if len(all_points) <= len(selected):
            break
        min_dists = []
        for p in all_points:
            dists = np.sqrt(np.sum((np.array(selected) - p)**2, axis=1))
            min_dists.append(np.min(dists))
        min_dists = np.array(min_dists)
        # Exclude already selected
        selected_set = set(map(tuple, selected))
        available = [i for i in range(len(all_points)) if tuple(all_points[i]) not in selected_set]
        if not available:
            break
        best_idx = available[np.argmax(min_dists[available])]
        selected.append(all_points[best_idx])
    
    grid = np.array(selected[:N])
    
    # Add small random perturbation to break symmetry
    perturbation = 0.01 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    
    return grid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initial Configuration - try multiple strategies
    centers = _build_initial_grid()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Try a few different initializations and keep the best
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        alt_centers = _build_initial_grid()
        alt_score = _compute_score(alt_centers)
        if alt_score > best_score:
            best_score = alt_score
            best_centers = alt_centers.copy()
            centers = alt_centers
    
    # 2. Dynamics Phase - run multiple times with different starting points
    dyn_time_budget = min(80.0, deadline - time.time() - 40.0)
    if dyn_time_budget > 15.0:
        # Run dynamics from best current
        dyn_centers = _dynamics_phase(centers, dyn_time_budget * 0.6)
        dyn_score = _compute_score(dyn_centers)
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dyn_centers.copy()
            centers = dyn_centers
        
        # Run dynamics from a perturbed version
        if time.time() < deadline - 30.0:
            perturbed = centers + 0.05 * np.random.randn(N, 3)
            perturbed = np.clip(perturbed, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
            dyn_centers2 = _dynamics_phase(perturbed, min(30.0, deadline - time.time() - 30.0))
            dyn_score2 = _compute_score(dyn_centers2)
            if dyn_score2 > best_score:
                best_score = dyn_score2
                best_centers = dyn_centers2.copy()
                centers = dyn_centers2
    
    # 3. Polish Phase with vectorized soft-min
    polish_time_budget = deadline - time.time() - 5.0
    
    if polish_time_budget > 5.0:
        # Schedule: start with moderate beta, increase more aggressively
        beta_schedule = [50, 200, 800, 3000, 10000, 30000, 80000]
        
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            
            try:
                start_params = centers.flatten()
                
                def fun(x):
                    val, _ = _soft_min_obj_and_grad(x, beta)
                    return val
                
                def jac(x):
                    _, grad = _soft_min_obj_and_grad(x, beta)
                    return grad
                
                # Use more iterations for higher beta
                max_iter = 150 if beta < 1000 else 200
                
                result = minimize(
                    fun,
                    start_params,
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100, 'maxcor': 50}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    centers = new_centers
            except Exception:
                pass
        
        # Final high-precision polish with very high beta
        if time.time() < deadline - 10.0:
            try:
                beta_final = 200000
                start_params = centers.flatten()
                
                def fun_final(x):
                    val, _ = _soft_min_obj_and_grad(x, beta_final)
                    return val
                
                def jac_final(x):
                    _, grad = _soft_min_obj_and_grad(x, beta_final)
                    return grad
                
                result = minimize(
                    fun_final,
                    start_params,
                    jac=jac_final,
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200, 'maxcor': 100}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
            except Exception:
                pass
    
    # Final validity check
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers

# EVOLVE-BLOCK-START
# The main construct() function is defined above
# EVOLVE-BLOCK-END
