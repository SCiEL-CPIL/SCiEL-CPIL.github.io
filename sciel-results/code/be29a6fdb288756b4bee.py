import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(pos):
    """Compute the maximum radius for a given set of positions."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Check bounds: centers must be in [-0.5, 0.5]
    if np.any(pos < -0.5) or np.any(pos > 0.5):
        return 0.0
        
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # For each point and each dimension, distance to nearest wall is 0.5 - |coord|
    wall_dists = 0.5 - np.abs(pos)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _energy_and_grad(pos_flat, target_r):
    """
    Compute energy and gradient for a target radius approach.
    Energy is sum of violations: max(0, target_r - dist/2)^2 for pairs and walls.
    We want to minimize this energy. If energy is 0, target_r is feasible.
    """
    pos = pos_flat.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(pos_flat)
    
    # Pairwise constraints
    # For each pair (i, j), we need dist(i,j) >= 2*target_r
    # Violation v_ij = max(0, 2*target_r - dist_ij)
    # Energy += v_ij^2
    # Grad w.r.t pos_i: 2 * v_ij * (pos_i - pos_j) / dist_ij
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    # Violations for pairs (i < j)
    # We only consider i < j to avoid double counting and self-interaction
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0.0, 2.0 * target_r - d_ij)
    
    energy += np.sum(v_ij ** 2)
    
    # Gradient for pairs
    # d/dpos_i (v^2) = 2v * dv/dpos_i
    # v = max(0, 2r - d)
    # dv/dpos_i = - (pos_i - pos_j) / d * (dv/dd)
    # dv/dd = -1 if violated, else 0
    # So dv/dpos_i = (pos_j - pos_i) / d if violated
    # d/dpos_i (v^2) = 2v * (pos_j - pos_i) / d
    
    # Only for violated pairs
    mask = v_ij > 0
    if np.any(mask):
        i_v = i_idx[mask]
        j_v = j_idx[mask]
        v_v = v_ij[mask]
        d_v = d_ij[mask]
        
        # Term for pos_i
        # grad_i += 2 * v * (pos_j - pos_i) / d
        term = 2.0 * v_v / d_v
        grad[i_v] += -term[:, np.newaxis] * diff[i_v] # diff[i] = pos_i - pos_j, so -term*diff = term*(pos_j-pos_i)
        # Wait: diff[i_v] = pos[i_v] - pos[j_v]
        # We want grad[pos_i] += 2*v * (pos_j - pos_i) / d
        # (pos_j - pos_i) = -diff[i_v]
        # So grad[pos_i] += -2*v/d * diff[i_v]
        # My code: grad[i_v] += -term[:, np.newaxis] * diff[i_v]
        # term = 2*v/d. So this is correct.
        
        # Term for pos_j
        # grad_j += 2 * v * (pos_i - pos_j) / d
        # (pos_i - pos_j) = diff[i_v]
        # So grad[pos_j] += term[:, np.newaxis] * diff[i_v]
        grad[j_v] += term[:, np.newaxis] * diff[i_v]

    # Wall constraints
    # For each point i and dim k, we need 0.5 - |pos_ik| >= target_r
    # i.e., pos_ik <= 0.5 - target_r AND pos_ik >= -(0.5 - target_r)
    # Let bound = 0.5 - target_r
    # Violation if pos_ik > bound: v = pos_ik - bound
    # Violation if pos_ik < -bound: v = -bound - pos_ik
    
    bound = 0.5 - target_r
    
    # Positive side: pos_ik > bound
    v_pos = np.maximum(0.0, pos - bound)
    energy += np.sum(v_pos ** 2)
    grad += 2.0 * v_pos
    
    # Negative side: pos_ik < -bound
    v_neg = np.maximum(0.0, -bound - pos)
    energy += np.sum(v_neg ** 2)
    grad += 2.0 * v_neg * (-1.0) # d/dx (-bound - x) = -1
    
    return energy, grad

def _optimize_for_radius(pos0, r_target, max_iter=500):
    """
    Try to find a configuration with radius at least r_target starting from pos0.
    Returns (success, pos).
    """
    pos0_flat = pos0.flatten()
    
    def objective(x):
        e, _ = _energy_and_grad(x, r_target)
        return e
    
    def gradient(x):
        _, g = _energy_and_grad(x, r_target)
        return g
        
    try:
        res = minimize(objective, pos0_flat, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        
        final_energy = res.fun
        # If energy is close to 0, we found a feasible solution for r_target
        if final_energy < 1e-8:
            pos_final = res.x.reshape(N, 3)
            # Clip to be safe
            pos_final = np.clip(pos_final, -0.5, 0.5)
            return True, pos_final
        else:
            return False, pos0
    except Exception:
        return False, pos0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    # Start with a reasonable initial guess
    # Grid-like initialization
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    # Add small random noise to break symmetry
    pos += np.random.normal(0, 0.01, (N, 3))
    pos = np.clip(pos, -0.49, 0.49)
    
    current_r = _compute_radius(pos)
    
    # Binary search on radius? Or incremental increase?
    # Incremental is more robust.
    # Start with current radius, try to increase it.
    
    r = current_r
    improvement_threshold = 1e-6
    
    while time.time() < deadline:
        # Try to increase radius
        # Step size decreases as we get closer to limit?
        # Let's try a fixed relative step
        step = max(0.001, 0.01 * r)
        r_new = r + step
        
        success, pos_new = _optimize_for_radius(pos, r_new)
        
        if success:
            # Check actual radius achieved
            actual_r = _compute_radius(pos_new)
            if actual_r > r:
                r = actual_r
                pos = pos_new
            else:
                # If we didn't improve, maybe step was too big or stuck
                # Reduce step? Or break?
                # Let's just keep pos if it's valid, but try smaller step next time?
                # For simplicity, if no improvement, we might be at a local optimum for this r.
                # Try a smaller step.
                step /= 2
                if step < 1e-6:
                    break
        else:
            # Failed to find feasible solution for r_new
            # Reduce step
            step /= 2
            if step < 1e-6:
                break
            
    # Final polishing: try to squeeze out more precision
    # Run a few more iterations with very small steps
    for _ in range(5):
        if time.time() > deadline:
            break
        r_new = r + 0.0001
        success, pos_new = _optimize_for_radius(pos, r_new)
        if success:
            actual_r = _compute_radius(pos_new)
            if actual_r > r:
                r = actual_r
                pos = pos_new
            else:
                break
        else:
            break
            
    return pos
