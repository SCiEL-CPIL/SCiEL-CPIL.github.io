import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    if r_pair < r_wall:
        return r_pair, 'pair', min_pair_idx, r_pair, r_wall
    else:
        return r_wall, 'wall', min_wall_idx, r_pair, r_wall

def _soft_min_objective_and_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    This approximates the minimum as beta -> infinity.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances (upper triangle only to save memory)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
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
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-14:
            vec = pos[i] - pos[j]
            coeff = w / (2.0 * d)
            grad_pos[i] += coeff * vec
            grad_pos[j] -= coeff * vec
    
    # Wall gradient
    for idx in range(N * 3):
        i = idx // 3
        k = idx % 3
        w = weights[len(pair_dists) + idx]
        s = np.sign(pos[i, k])
        if s == 0:
            s = 1.0
        grad_pos[i, k] -= w * s
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    return -soft_min_val, -grad_pos.flatten()

def _soft_min_optimize(pos, beta=100.0, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Optimize using soft-min objective."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta)[0],
            pos.flatten(),
            jac=lambda x: _soft_min_objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _energy_objective_and_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
    This is C^1 smooth and aggregates all violations.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Compute violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    wall_viol = np.maximum(0.0, r_target - d_walls)
    
    energy = 0.5 * np.sum(pair_viol**2) + 0.5 * np.sum(wall_viol**2)
    
    # Gradient - vectorized
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient: for each violating pair, push apart
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol_vals = pair_viol[iu]
    
    mask = pair_viol_vals > 0
    if np.any(mask):
        i_idx = iu[0][mask]
        j_idx = iu[1][mask]
        d_vals = pair_dists[mask]
        v_vals = pair_viol_vals[mask]
        
        # Avoid division by zero
        valid = d_vals > 1e-14
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            d_v = d_vals[valid]
            v_v = v_vals[valid]
            
            vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
            coeffs = v_v / d_v  # shape (n_valid,)
            
            # Accumulate gradients
            np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
            np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    wall_viol_mask = wall_viol > 0
    if np.any(wall_viol_mask):
        i_idx = np.where(wall_viol_mask)[0]
        k_idx = np.where(wall_viol_mask)[1]
        v_vals = wall_viol[wall_viol_mask]
        
        signs = np.sign(pos[i_idx, k_idx])
        signs[signs == 0] = 1.0
        
        grad_pos[i_idx, k_idx] += v_vals * signs
    
    return energy, -grad_pos.flatten()

def _energy_minimize(pos, r_target, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for a fixed target radius."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_objective_and_grad(x, r_target)[0],
            pos.flatten(),
            jac=lambda x: _energy_objective_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 1e-8:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _radius_increase_search(pos, r_start, r_end, n_steps=50, deadline=None):
    """
    Iteratively increase target radius and minimize energy at each step.
    This escapes local minima of the hard-min objective.
    """
    r_values = np.linspace(r_start, r_end, n_steps)
    current_pos = pos.copy()
    
    for r in r_values:
        if deadline is not None and time.time() > deadline:
            break
        current_pos = _energy_minimize(current_pos, r, max_iter=100, ftol=1e-12, gtol=1e-10)
    
    return current_pos

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass using both soft-min and energy-based methods."""
    # First try soft-min optimization
    polished = _soft_min_optimize(pos, beta=200.0, max_iter=max_iter, ftol=ftol, gtol=gtol)
    val1, _, _, _, _ = compute_min_dist(polished)
    
    # Then try energy-based polish
    val, _, _, _, _ = compute_min_dist(polished)
    r_target = val * 1.001
    polished2 = _energy_minimize(polished, r_target, max_iter=max_iter, ftol=ftol, gtol=gtol)
    val2, _, _, _, _ = compute_min_dist(polished2)
    
    # Return the better one
    if val2 > val1:
        return polished2
    return polished

def construct():
    global rng_global
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    rng_global = np.random.default_rng(42)
    
    # Initial guess: Better grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    # Evaluate initial
    val, _, _, _, _ = compute_min_dist(best_pos)
    best_r = val
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Phase 1: Energy-based radius increase from multiple starting points
    phase1_deadline = start_time + 80.0
    attempt = 0
    
    while time.time() < phase1_deadline:
        attempt += 1
        
        # Choose perturbation strategy
        if attempt % 10 == 0:
            # Random restart
            start_pos = rng_global.uniform(-0.45, 0.45, size=(N, 3))
        elif attempt % 10 == 1:
            # Small perturbation
            scale = 0.01
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 2:
            # Medium perturbation
            scale = 0.04
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 3:
            # Large perturbation
            scale = 0.08
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 4:
            # Very small perturbation
            scale = 0.003
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 5:
            # Swap-based perturbation: swap two random spheres
            start_pos = best_pos.copy()
            i, j = rng_global.choice(N, size=2, replace=False)
            start_pos[[i, j]] = start_pos[[j, i]]
            start_pos += rng_global.normal(0, 0.01, size=(N, 3))
        else:
            # Kick: randomly displace a few spheres
            start_pos = best_pos.copy()
            n_kick = rng_global.integers(3, 10)
            kick_idx = rng_global.choice(N, size=n_kick, replace=False)
            start_pos[kick_idx] += rng_global.normal(0, 0.05, size=(n_kick, 3))
        
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Energy-based radius increase
        # Start from a radius below current best to allow exploration
        r_start = best_r * 0.85
        r_end = best_r * 1.05
        
        new_pos = _radius_increase_search(start_pos, r_start, r_end, n_steps=30, deadline=phase1_deadline)
        
        # Also try soft-min optimization on the result
        new_pos_soft = _soft_min_optimize(new_pos, beta=100.0, max_iter=200, ftol=1e-12, gtol=1e-10)
        val_soft, _, _, _, _ = compute_min_dist(new_pos_soft)
        
        # Evaluate energy-based result
        val, _, _, _, _ = compute_min_dist(new_pos)
        
        # Use the better of the two
        if val_soft > val:
            new_pos = new_pos_soft
            val = val_soft
        
        # If improved, try gradient refinement
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
            
            # Polish with gradient method
            if time.time() < phase1_deadline - 5.0:
                polished = _polish(best_pos, max_iter=300, ftol=1e-13, gtol=1e-11)
                val2, _, _, _, _ = compute_min_dist(polished)
                if val2 > best_r:
                    best_r = val2
                    best_pos = polished
    
    # Phase 2: Intensive energy-based refinement around best
    phase2_deadline = deadline - 15.0
    while time.time() < phase2_deadline:
        # Small perturbations around best
        scale = 0.005
        start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Energy-based radius increase with tighter range
        r_start = best_r * 0.98
        r_end = best_r * 1.02
        
        new_pos = _radius_increase_search(start_pos, r_start, r_end, n_steps=20, deadline=phase2_deadline)
        
        # Also try soft-min
        new_pos_soft = _soft_min_optimize(new_pos, beta=200.0, max_iter=200, ftol=1e-12, gtol=1e-10)
        val_soft, _, _, _, _ = compute_min_dist(new_pos_soft)
        
        val, _, _, _, _ = compute_min_dist(new_pos)
        
        if val_soft > val:
            new_pos = new_pos_soft
            val = val_soft
        
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=500, ftol=1e-13, gtol=1e-11)
        val, _, _, _, _ = compute_min_dist(polished_pos)
        if val > best_r:
            best_r = val
            best_pos = polished_pos
    
    # Ensure valid output
    best_pos = np.clip(best_pos, -HALF, HALF)
    
    return best_pos

# EVOLVE-BLOCK-END
