"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_radius(x, beta=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns (negative soft_min, gradient) for minimization.
    Vectorized gradient computation.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraints into a single array
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min via log-sum-exp
    min_d = np.min(all_dists)
    if min_d < 1e-10:
        return 0.0, np.zeros_like(x)
    
    shifted = all_dists - min_d
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    soft_min = min_d - np.log(sum_exp) / beta
    
    # Gradient: d(soft_min)/dx = -sum(w_i * d(d_i)/dx) / sum_exp
    # where w_i = exp(-beta * (d_i - min_d))
    # We want to minimize -soft_min, so grad = sum(w_i * d(d_i)/dx) / sum_exp
    
    weights = exp_terms / sum_exp  # normalized weights
    
    grad = np.zeros((n, 3))
    
    # Pairwise gradient contributions
    # d(pair_dist_ij)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    # d(pair_dist_ij)/d(c_j) = (c_j - c_i) / (2 * d_ij)
    pair_weights = weights[:len(pair_dists)]
    d_ij = dists[i_idx, j_idx]
    valid = d_ij > 1e-10
    
    for k in range(len(pair_dists)):
        if not valid[k] or pair_weights[k] < 1e-15:
            continue
        i = i_idx[k]
        j = j_idx[k]
        w = pair_weights[k]
        vec = (centers[i] - centers[j]) / (2.0 * d_ij[k])
        grad[i] += w * vec
        grad[j] -= w * vec
    
    # Wall gradient contributions
    # d(wall_dist_i)/d(c_i) = -sign(c_i[k]) where k is argmax |c_i|
    wall_weights = weights[len(pair_dists):]
    for i in range(n):
        if wall_weights[i] < 1e-15:
            continue
        max_val = max_abs[i]
        w = wall_weights[i]
        for k in range(3):
            if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                grad[i, k] -= w * np.sign(centers[i, k])
                break
    
    # We're minimizing -soft_min, so negate the gradient
    grad = -grad.flatten()
    
    return -soft_min, grad

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Uses vectorized operations for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Vectorized computation for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        # Gradient: for each violating pair, add force
        # dE/dc_i = -2 * v_ij * (c_i - c_j) / d_ij
        # dE/c_j = 2 * v_ij * (c_i - c_j) / d_ij
        for k in range(len(v_ij)):
            if v_ij[k] <= 0:
                continue
            i = i_idx[k]
            j = j_idx[k]
            v = v_ij[k]
            d = d_ij[k]
            if d < 1e-10:
                continue
            factor = -2.0 * v / d
            vec = centers[i] - centers[j]
            force = factor * vec
            grad[i] += force
            grad[j] -= force

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        for i in range(n):
            if v_wall[i] <= 0:
                continue
            v = v_wall[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    High-precision polish pass to maximize the minimum radius.
    Uses a smooth objective with tight tolerances and adaptive beta.
    """
    start_time = time.time()
    
    def objective_and_grad(x, beta=200.0):
        val, g = _soft_min_radius(x, beta=beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_polished = centers.copy()
    best_r = _compute_radius(centers)
    
    # Try multiple beta values for the soft-min, increasing for sharper approximation
    for beta_val in [100.0, 200.0, 400.0]:
        if time.time() > start_time + time_budget:
            break
        
        def obj_grad(x):
            return objective_and_grad(x, beta=beta_val)
        
        try:
            res = minimize(
                obj_grad,
                best_polished.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000, 'maxls': 50}
            )
            
            polished = res.x.reshape(-1, 3)
            polished = np.clip(polished, -0.4999999, 0.4999999)
            r_polished = _compute_radius(polished)
            
            if r_polished > best_r:
                best_r = r_polished
                best_polished = polished.copy()
                
        except Exception:
            continue
    
    return best_polished

def _stochastic_local_search(centers, time_budget=15.0, rng_seed=42):
    """
    Stochastic local search with simulated annealing to escape local minima.
    Dynamically swaps positions of overlapping spheres and applies small random perturbations.
    Accepts uphill moves based on a temperature schedule to avoid getting stuck.
    """
    start_time = time.time()
    rng = np.random.RandomState(rng_seed)
    
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    # Temperature schedule
    T_initial = 0.05
    T_final = 0.0001
    T = T_initial
    
    # Precompute indices for efficiency
    n = len(current_centers)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > start_time + time_budget:
            break
        
        # Update temperature (geometric cooling)
        T = T_initial * (T_final / T_initial) ** (it / max_iter)
        
        # Choose move type: swap, perturb, or both
        move_type = rng.random()
        
        if move_type < 0.3:
            # Swap two centers
            i, j = rng.choice(n, size=2, replace=False)
            proposed = current_centers.copy()
            proposed[i], proposed[j] = current_centers[j], current_centers[i]
        elif move_type < 0.7:
            # Perturb one center
            i = rng.randint(n)
            delta = rng.normal(0, 0.02, size=3)
            proposed = current_centers.copy()
            proposed[i] += delta
            # Keep within bounds
            proposed[i] = np.clip(proposed[i], -0.49999, 0.49999)
        else:
            # Perturb two centers (swap-like with noise)
            i, j = rng.choice(n, size=2, replace=False)
            delta1 = rng.normal(0, 0.015, size=3)
            delta2 = rng.normal(0, 0.015, size=3)
            proposed = current_centers.copy()
            proposed[i] += delta1
            proposed[j] += delta2
            # Keep within bounds
            proposed[i] = np.clip(proposed[i], -0.49999, 0.49999)
            proposed[j] = np.clip(proposed[j], -0.49999, 0.49999)
        
        # Compute new radius
        proposed_radius = _compute_radius(proposed)
        
        # Acceptance criterion (simulated annealing)
        delta_r = proposed_radius - current_radius
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_centers = proposed
            current_radius = proposed_radius
            
            # Update best
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
        else:
            # Reject move, but occasionally try a larger perturbation to escape
            if rng.random() < 0.05:
                # Large random perturbation
                n_perturb = rng.randint(3, 10)
                indices = rng.choice(n, size=n_perturb, replace=False)
                proposed = current_centers.copy()
                for idx in indices:
                    proposed[idx] += rng.normal(0, 0.03, size=3)
                    proposed[idx] = np.clip(proposed[idx], -0.49999, 0.49999)
                
                proposed_radius = _compute_radius(proposed)
                if proposed_radius > current_radius * 0.95:  # Allow slight degradation
                    current_centers = proposed
                    current_radius = proposed_radius
    
    return best_centers

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_budget=5.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    start_time = time.time()
    
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
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > time_budget:
             pass
            
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # Multiple grid configurations
    grid_configs = [
        (5, 4, 5),  # 100
        (4, 5, 5),  # 100
        (5, 5, 4),  # 100
        (10, 10, 1),  # 100 (flat)
        (1, 10, 10),  # 100 (flat)
        (10, 1, 10),  # 100 (flat)
    ]
    
    rng = np.random.RandomState(42)
    
    # Create a set of initial guesses
    inits = []
    for nx, ny, nz in grid_configs:
        if nx * ny * nz == N:
            inits.append(make_grid(nx, ny, nz))
    
    # Add random inits
    for i in range(8):
        rc = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rc)
    
    # Add structured random: start from grid, add noise
    base_grid = make_grid(5, 4, 5)
    for i in range(4):
        noise_scale = 0.02 * (i + 1)
        noisy = base_grid + rng.normal(0, noise_scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        
        # 2. Iteratively increase target radius
        target_R = max(r_est * 1.005, 0.05)
        
        current_centers = init.copy()
        max_iter_per_step = 100
        
        stuck_count = 0
        while time.time() < deadline - 15.0:
            # Try to optimize for target_R
            time_budget_for_step = min(8.0, max(2.0, (deadline - 10.0 - time.time())))
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=max_iter_per_step, time_budget=time_budget_for_step)
            
            # Check if we achieved a valid packing for this radius
            if energy < 1e-8:
                r_achieved = _compute_radius(optimized_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = optimized_centers.copy()
                
                # Try to increase the radius
                increment = 0.003 if target_R < 0.11 else 0.001
                target_R *= (1.0 + increment)
                current_centers = optimized_centers
                stuck_count = 0
            else:
                # Failed to fit this radius
                r_curr = _compute_radius(current_centers)
                if r_curr > best_radius:
                    best_radius = r_curr
                    best_centers = current_centers.copy()
                
                stuck_count += 1
                
                # Try to heal with slightly smaller radius
                heal_attempts = 0
                heal_R = target_R * 0.98
                healed = False
                while heal_attempts < 5 and time.time() < deadline - 15.0:
                    h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50)
                    if h_energy < 1e-6:
                        current_centers = h_centers
                        target_R = heal_R * 1.002
                        healed = True
                        break
                    else:
                        heal_R *= 0.98
                        heal_attempts += 1
                        if h_energy > 1.0:
                            break
                
                if not healed or stuck_count > 5:
                    break
        
        # Update best with current optimized centers
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()
        
        # Apply stochastic local search to escape local minima
        if time.time() < deadline - 25.0:
            sls_budget = min(10.0, deadline - 25.0 - time.time())
            if sls_budget > 2.0:
                sls_centers = _stochastic_local_search(current_centers, time_budget=sls_budget, rng_seed=int(time.time() * 1000) % 100000)
                r_sls = _compute_radius(sls_centers)
                if r_sls > best_radius:
                    best_radius = r_sls
                    best_centers = sls_centers.copy()

    # Final high-precision polish pass with extended budget
    if best_centers is not None and time.time() < deadline - 5.0:
        polish_time = min(30.0, deadline - 5.0 - time.time())
        if polish_time > 2.0:
            polished = _polish_centers(best_centers, max_iter=500, time_budget=polish_time)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
    
    # Additional stochastic local search pass with different seed
    if best_centers is not None and time.time() < deadline - 3.0:
        sls_time2 = min(5.0, deadline - 3.0 - time.time())
        if sls_time2 > 1.0:
            sls2 = _stochastic_local_search(best_centers, time_budget=sls_time2, rng_seed=12345)
            r_sls2 = _compute_radius(sls2)
            if r_sls2 > best_radius:
                best_radius = r_sls2
                best_centers = sls2
    
    # Second polish after SLS if time permits
    if best_centers is not None and time.time() < deadline - 2.0:
        polish_time2 = min(15.0, deadline - 2.0 - time.time())
        if polish_time2 > 2.0:
            polished2 = _polish_centers(best_centers, max_iter=400, time_budget=polish_time2)
            r_polished2 = _compute_radius(polished2)
            if r_polished2 > best_radius:
                best_radius = r_polished2
                best_centers = polished2
    
    # Final safety check
    if best_centers is None:
        best_centers = make_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        # Ensure all values are finite
        if not np.all(np.isfinite(best_centers)):
            best_centers = make_grid(5, 4, 5)
        
    return best_centers
# EVOLVE-BLOCK-END
