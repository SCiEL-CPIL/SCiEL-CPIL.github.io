"""
Strategy: Simulated Annealing on the Sphere with Riemannian Moves.
Instead of pure gradient descent (L-BFGS-B), we use Simulated Annealing (SA) 
to escape local minima. The state is a set of N points on the unit sphere.
Moves are small random rotations of individual points or small groups of points.
The acceptance criterion is Metropolis: accept with probability 1 if energy decreases, 
else with probability exp(-delta_E / T).
We use a cooling schedule that allows exploring the landscape broadly at first 
(high T) and then refining (low T).
To handle the 120s limit, we run multiple independent SA chains with different 
random seeds and initial configurations (Fibonacci sphere, random), keeping the 
best result found.
We also incorporate a "local polish" step using L-BFGS-B at the end of each SA 
run or periodically, to ensure we reach the bottom of the local basin found by SA.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns energy only.
    """
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf')
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    return energy

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy
    energy = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw_vec = grad_tangent / norms
    
    return energy, grad_raw_vec.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _simulate_annealing(pts_init, start_time, deadline, seed):
    """
    Run Simulated Annealing for a given initial configuration.
    """
    np.random.seed(seed)
    pts = pts_init.copy()
    current_energy = _log_energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Cooling schedule parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.995  # Cooling rate
    T = T_start
    
    # Move parameters
    step_size = 0.05
    
    while T > T_end and time.time() < deadline:
        # Select a random point to move
        idx = np.random.randint(N)
        
        # Generate a random rotation axis
        axis = np.random.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            continue
        axis /= axis_norm
        
        # Random angle
        angle = np.random.uniform(-step_size, step_size)
        
        # Rotation matrix (Rodrigues)
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        
        # Apply rotation to the selected point
        new_pt = R @ pts[idx]
        
        # Calculate energy difference
        # We can approximate the energy change by only recalculating terms involving idx
        # E = sum_{i<j} log(1/d_ij)
        # Delta E = (sum_{j!=idx} log(1/d_new_idx_j)) - (sum_{j!=idx} log(1/d_old_idx_j))
        
        old_pt = pts[idx]
        
        # Distances from old point to all others
        diff_old = pts - old_pt
        diff_old[idx] = 0 # Avoid self
        dist_sq_old = np.sum(diff_old * diff_old, axis=1)
        dist_sq_old[idx] = 1.0
        # Filter out zeros/invalids just in case, though idx is set to 1.0
        valid_old = dist_sq_old > 1e-15
        log_terms_old = -0.5 * np.log(dist_sq_old[valid_old])
        sum_old = np.sum(log_terms_old)
        
        # Distances from new point to all others
        diff_new = pts - new_pt
        diff_new[idx] = 0
        dist_sq_new = np.sum(diff_new * diff_new, axis=1)
        dist_sq_new[idx] = 1.0
        valid_new = dist_sq_new > 1e-15
        log_terms_new = -0.5 * np.log(dist_sq_new[valid_new])
        sum_new = np.sum(log_terms_new)
        
        delta_E = sum_new - sum_old
        
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            pts[idx] = new_pt
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        # Cool down
        T *= alpha
        
        # Periodically check time and maybe adjust step size
        if time.time() > deadline:
            break

    return best_pts, best_energy

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Fibonacci
    fib_pts = _fibonacci_sphere(N)
    candidates.append(fib_pts)
    
    # 2. Random (a few different seeds)
    for seed in [42, 123, 999]:
        rand_pts = _random_uniform_sphere(N, seed)
        candidates.append(rand_pts)
        
    best_energy = float('inf')
    best_pts = None
    
    # Run SA for each candidate
    # We divide the time budget among the candidates
    # Let's give each candidate about 20-30 seconds of SA time, plus polish time
    time_per_candidate = 20.0
    
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:  # Leave 5s for final steps
            break
            
        # Run SA
        sa_deadline = start_time + elapsed + time_per_candidate
        sa_deadline = min(sa_deadline, DEADLINE - 10) # Leave 10s for polish
        
        try:
            sa_pts, sa_energy = _simulate_annealing(init_pts, start_time, sa_deadline, seed=1000+i)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts
                
        except Exception:
            pass # Ignore failures for this start
        
    # If we still have time, do a final polish on the best result using L-BFGS-B
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            init_y = best_pts.flatten()
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 200,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _objective_and_grad(pts_opt.flatten())
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback if something went wrong
    if best_pts is None:
        best_pts = fib_pts

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
