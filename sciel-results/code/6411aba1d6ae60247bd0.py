import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
DEADLINE_MARGIN = 10.0  # seconds

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def _soft_min_obj_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    Returns (objective_value, gradient).
    Objective is -soft_min, so minimizing it maximizes soft_min.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    pair_dists = dists[i_idx, j_idx]
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # All effective distances (pair distances divided by 2, wall distances as-is)
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -logsumexp(-beta * x) / beta
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. positions
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient
    n_pairs = len(i_idx)
    pair_weights = weights[:n_pairs]
    
    # Avoid division by zero
    valid = pair_dists > 1e-14
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = pair_dists[valid]
        w_v = pair_weights[valid]
        
        vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
        coeffs = w_v / (2.0 * d_v) # shape (n_valid,)
        
        # grad_pos[i] += coeff * vec
        np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
        # grad_pos[j] -= coeff * vec
        np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    signs[signs == 0] = 1.0
    
    # grad_pos[i, k] += -w_k * sign(pos_ik)
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    # Gradient of -soft_min is -grad(soft_min)
    return -soft_min_val, -grad_pos.flatten()

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # 2. Wall terms (vectorized)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        # For each violating sphere, the gradient points away from the nearest wall
        max_k = np.argmax(abs_centers[i_valid_wall], axis=1)
        signs = np.sign(centers[i_valid_wall, max_k])
        grad_vals = 2.0 * violations_wall * signs
        
        # Scatter the gradient contributions
        np.add.at(grad, (i_valid_wall, max_k), grad_vals)
        
    return energy, grad.flatten()

def _minimize_soft_min(init_centers, beta, max_iter=300, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the negative soft-min objective.
    """
    def objective_and_grad(x):
        val, g = _soft_min_obj_grad(x, beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers, res.success
    except Exception:
        return init_centers, False

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """
    High-precision polish pass using the hard-min objective via energy minimization
    with a tight target radius.
    """
    # First get current radius
    r_current = compute_min_dist(pos)
    
    # Try to squeeze by minimizing energy at r_current + small epsilon
    # This effectively pushes the configuration to expand
    for _ in range(3):
        # Try slightly larger radius
        r_try = r_current * (1.0 + 1e-6)
        new_pos, energy, success = _minimize_energy(pos, r_try, max_iter=max_iter, ftol=ftol, gtol=gtol)
        
        if success and energy < 1e-8:
            # Check if it actually improved
            r_new = compute_min_dist(new_pos)
            if r_new > r_current:
                pos = new_pos
                r_current = r_new
            else:
                break
        else:
            break
    
    return pos

def _generate_initial_positions():
    """
    Generate initial positions using a grid structure.
    """
    # 5x5x4 grid = 100 points
    # Or 4x5x5
    # Let's use a slightly perturbed grid
    m1, m2, m3 = 5, 5, 4
    x = np.linspace(-0.4, 0.4, m1)
    y = np.linspace(-0.4, 0.4, m2)
    z = np.linspace(-0.4, 0.4, m3)
    
    grid = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 140s deadline
    
    # Initial guess
    best_pos = _generate_initial_positions()
    best_r = compute_min_dist(best_pos)
    
    rng = np.random.default_rng(42)
    
    attempts = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        attempts += 1
        
        # Decide start point
        if attempts % 5 == 0:
            # Random start
            start_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        elif attempts % 5 == 1:
            # Small perturbation of best
            scale = 0.02
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 2:
            # Medium perturbation
            scale = 0.05
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        elif attempts % 5 == 3:
            # Large perturbation
            scale = 0.1
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        else:
            # Very small perturbation for fine-tuning
            scale = 0.005
            start_pos = best_pos + rng.normal(0, scale, size=(N, 3))
        
        # Clip to stay inside bounds
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Strategy: Alternate between soft-min (global) and energy minimization (local refinement)
        # Use soft-min for most attempts to escape local minima, energy for polishing
        use_soft = (attempts % 3 != 0)  # 2/3 soft, 1/3 energy
        
        try:
            if use_soft:
                # Adaptive beta
                beta = 50.0 + 50.0 * (attempts % 5)
                new_pos, success = _minimize_soft_min(start_pos, beta, max_iter=100, ftol=1e-10, gtol=1e-8)
            else:
                # Energy minimization with target radius slightly above current best
                target_R = best_r * (1.0 + 0.01)
                new_pos, energy, success = _minimize_energy(start_pos, target_R, max_iter=100, ftol=1e-10, gtol=1e-8)
            
            if success:
                new_pos = np.clip(new_pos, -0.5, 0.5)
                if np.all(np.isfinite(new_pos)):
                    val = compute_min_dist(new_pos)
                    if val > best_r:
                        best_r = val
                        best_pos = new_pos.copy()
                        last_improvement_time = time.time()
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=300, ftol=1e-12, gtol=1e-10)
        val = compute_min_dist(polished_pos)
        if val > best_r:
            best_r = val
            best_pos = polished_pos
    
    return best_pos
