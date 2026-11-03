"""Combined program: Energy-based packing with SGLD, soft-min polishing, and BCC initialization."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """Compute the maximum possible radius for a set of centers C."""
    if len(C) < 2:
        return 0.5 - np.max(np.abs(C))
    
    # Boundary constraint
    dist_to_bound = 0.5 - np.abs(C).max(axis=1)
    r_bound = np.min(dist_to_bound)
    
    # Pairwise constraint
    D = cdist(C, C)
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    r_pair = min_dist / 2.0
    
    return min(r_bound, r_pair)

def _generate_bcc_points(n_max=4):
    """Generate a set of BCC lattice points centered at the origin."""
    points = []
    # Corner points
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                points.append([i, j, k])
    # Body center points
    for i in range(-n_max, n_max):
        for j in range(-n_max, n_max):
            for k in range(-n_max, n_max):
                points.append([i + 0.5, j + 0.5, k + 0.5])
    
    points = np.array(points)
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:N]]

def _energy_objective_and_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
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

def _soft_min_objective_and_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
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

def _sgld_step(pos, r_target, temperature, step_size, rng):
    """
    Perform one Stochastic Gradient Langevin Dynamics step.
    Uses full gradient for stability, but adds noise to escape local minima.
    """
    # Compute gradient
    _, grad_flat = _energy_objective_and_grad(pos.flatten(), r_target)
    grad = grad_flat.reshape(N, 3)
    
    # SGLD update: x_{t+1} = x_t + step_size * grad + sqrt(2 * step_size * temperature) * noise
    noise = rng.normal(size=pos.shape)
    noise_scale = np.sqrt(2.0 * step_size * temperature)
    
    new_pos = pos + step_size * grad + noise_scale * noise
    
    # Project back to box [-0.5, 0.5]
    new_pos = np.clip(new_pos, -HALF, HALF)
    
    return new_pos

def _sgld_optimize(pos, r_target, initial_temp=0.01, final_temp=0.001, steps=200, rng=None):
    """
    Run SGLD optimization with a cooling schedule.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_pos = pos.copy()
    temps = np.linspace(initial_temp, final_temp, steps)
    base_step = 0.01
    
    for t in range(steps):
        if _check_time():
            break
            
        temp = temps[t]
        # Step size can be slightly reduced as temperature drops
        step_size = base_step * (1.0 - 0.5 * (t / steps))
        
        current_pos = _sgld_step(current_pos, r_target, temp, step_size, rng)
        
        # Periodically project or check validity? 
        # The clip in _sgld_step ensures boundary constraints.
        # We don't explicitly check pair overlaps here, the energy gradient handles it.
        
    return current_pos

def _radius_increase_search(pos, r_start, r_end, n_steps=50, deadline=None):
    """
    Iteratively increase target radius and minimize energy at each step.
    """
    r_values = np.linspace(r_start, r_end, n_steps)
    current_pos = pos.copy()
    
    for r in r_values:
        if deadline is not None and time.time() > deadline:
            break
        if _check_time():
            break
        current_pos = _energy_minimize(current_pos, r, max_iter=100, ftol=1e-12, gtol=1e-10)
    
    return current_pos

def construct():
    global START_TIME
    START_TIME = time.time() # Reset timer
    
    # 1. Generate base lattice points
    V = _generate_bcc_points(4)
    
    # 2. Scale and translate to fit in cube
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    V_extent = V_max - V_min
    max_extent = np.max(V_extent)
    
    # Estimate scale s
    s_init = 1.0 / (0.866 + max_extent)
    t_init = np.array([0.0, 0.0, 0.0])
    
    C = s_init * V + t_init
    
    # 3. Initial Energy Minimization / Radius Increase
    # Estimate a starting radius
    r_current = _compute_radius(C)
    
    # Try to increase radius using energy minimization
    # We aim for slightly above the known best (0.11138)
    # But start from current best estimate.
    
    # First, do a quick polish with soft-min to get a better starting point
    C = _soft_min_optimize(C, beta=50.0, max_iter=200)
    r_current = _compute_radius(C)
    
    # Now, do a radius increase search
    # Start from r_current, go up to 0.115 (slightly above best known)
    # If we can't reach 0.115, we'll settle for the best we can.
    r_target_max = 0.115
    C = _radius_increase_search(C, r_current, r_target_max, n_steps=30)
    
    # 4. SGLD Phase
    # Use SGLD to escape local minima.
    # We target the best known radius or slightly higher.
    # Let's target 0.112 (slightly above seed, below best known) to be safe, 
    # or just use the current best radius as target.
    
    best_r = _compute_radius(C)
    best_C = C.copy()
    
    # Run SGLD for a few iterations
    # We'll run SGLD targeting a radius slightly higher than current best
    # to push the system towards higher density.
    
    rng = np.random.default_rng(123)
    
    # Run SGLD for multiple restarts or just one long run?
    # One long run with cooling is usually better.
    
    # Target radius: let's try to push it to 0.113
    r_target_sgl = min(0.113, best_r * 1.05) # Don't jump too high
    
    # Actually, if best_r is already 0.111, targeting 0.113 is a good push.
    # If best_r is low, targeting 0.113 might be too hard.
    # Let's target max(best_r + 0.001, 0.112)
    r_target_sgl = max(best_r + 0.001, 0.112)
    
    C_sgl = _sgld_optimize(C, r_target_sgl, initial_temp=0.005, final_temp=0.0005, steps=300, rng=rng)
    
    r_sgl = _compute_radius(C_sgl)
    if r_sgl > best_r:
        best_r = r_sgl
        best_C = C_sgl.copy()
    
    # 5. Final Polish
    # Use soft-min for high precision
    C_polished = _soft_min_optimize(best_C, beta=200.0, max_iter=500)
    r_polished = _compute_radius(C_polished)
    
    if r_polished > best_r:
        best_r = r_polished
        best_C = C_polished.copy()
    
    # One more energy minimize at the final radius to ensure no violations
    C_final = _energy_minimize(best_C, best_r * 0.999, max_iter=100)
    r_final = _compute_radius(C_final)
    
    if r_final > best_r:
        best_r = r_final
        best_C = C_final.copy()
    
    # Safety check: ensure validity
    if not np.all(np.isfinite(best_C)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    # Ensure inside cube (clip just in case)
    best_C = np.clip(best_C, -HALF + 1e-9, HALF - 1e-9)
    
    return best_C
