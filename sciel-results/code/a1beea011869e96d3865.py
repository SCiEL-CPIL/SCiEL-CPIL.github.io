"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

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
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

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

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _soft_min_objective_and_grad(x, beta=100.0):
    """
    Smooth surrogate for maximizing the minimum distance using soft-min.
    Maximizes soft_min(min_pair_dist/2, min_wall_dist).
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # All constraints: pairwise/2 and wall distances
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    all_constraints = np.concatenate([pair_dists, wall_dists])
    
    # Soft-min: -log(sum(exp(-beta * x))) / beta
    # Gradient: d/dx_i = sum_j (exp(-beta * x_j) / sum_k exp(-beta * x_k)) * (-beta * dx_j/dx_i)
    scaled = -beta * all_constraints
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    # Objective: soft_min = -log(sum_exp + exp(max_scaled)) / beta
    # Actually: soft_min = -(log(sum(exp(scaled))) - max_scaled) / beta = -(log(sum_exp) + max_scaled) / beta
    objective = -(np.log(sum_exp) + max_scaled) / beta
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    pair_weights = weights[:len(pair_dists)]
    if len(pair_weights) > 0:
        for idx in range(len(pair_weights)):
            w = pair_weights[idx]
            i = i_idx[idx]
            j = j_idx[idx]
            d = dists[i, j]
            if d > 1e-12:
                vec = centers[i] - centers[j]
                grad[i] += beta * w * vec / d
                grad[j] -= beta * w * vec / d
    
    # Wall gradient
    wall_weights = weights[len(pair_dists):]
    for idx in range(len(wall_weights)):
        w = wall_weights[idx]
        i = idx
        max_k = np.argmax(abs_centers[i])
        grad[i, max_k] -= beta * w * np.sign(centers[i, max_k])
    
    return objective, grad.flatten()

def _repulsive_gas_energy_and_grad(x, R, softness=50.0):
    """
    Repulsive gas model: smooth penalty for distances < 2R.
    Uses smooth approximation to avoid discontinuities.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise: smooth penalty for d < 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Smooth step: f(d) = 2R - d if d < 2R, else 0
    # Use softplus-like: f(d) = log(1 + exp(softness*(2R-d))) / softness
    x_val = softness * (2.0 * R - d_ij)
    # Clip for numerical stability
    x_val = np.clip(x_val, -50, 50)
    soft_viol = np.log1p(np.exp(x_val)) / softness
    
    energy += np.sum(soft_viol)
    
    # Gradient: d/dx_i = -softness * sigmoid(softness*(2R-d)) * (x_i - x_j) / d
    sigmoid_vals = 1.0 / (1.0 + np.exp(-x_val))
    factors = sigmoid_vals / d_ij
    
    vecs = centers[i_idx] - centers[j_idx]
    forces = (factors * 1.0)[:, np.newaxis] * vecs  # Note: gradient of log(1+exp(x)) is sigmoid(x)
    
    # dE/dx_i = -sum_j sigmoid(softness*(2R-d_ij)) * (x_i - x_j) / d_ij
    np.add.at(grad, i_idx, -forces)
    np.add.at(grad, j_idx, forces)
    
    # Wall terms: smooth penalty for wall_dist < R
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    x_wall = softness * (R - wall_dists)
    x_wall = np.clip(x_wall, -50, 50)
    soft_wall_viol = np.log1p(np.exp(x_wall)) / softness
    
    energy += np.sum(soft_wall_viol)
    
    sigmoid_wall = 1.0 / (1.0 + np.exp(-x_wall))
    for idx in range(n):
        max_k = np.argmax(abs_centers[idx])
        grad[idx, max_k] -= sigmoid_wall[idx] * np.sign(centers[idx, max_k])
    
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius using L-BFGS-B.
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

def _simulated_annealing_step(centers, R, T, rng, n_steps=50, step_scale=0.01):
    """
    Simulated annealing step to escape local minima.
    """
    current = centers.copy()
    current_energy, _ = _energy_and_gradient(current, R)
    
    for _ in range(n_steps):
        # Propose random perturbation
        perturbation = rng.normal(0, step_scale, size=current.shape)
        proposed = current + perturbation
        proposed = np.clip(proposed, -0.49, 0.49)
        
        proposed_energy, _ = _energy_and_gradient(proposed, R)
        
        # Metropolis criterion
        delta_E = proposed_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
            current = proposed
            current_energy = proposed_energy
    
    return current

def _polish_with_high_precision(centers, target_R, max_iter=2000, ftol=1e-18, gtol=1e-16):
    """
    High-precision polish pass.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers
    except Exception:
        return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization: 5x4x5 Grid with symmetry breaking
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    # Add small noise to break symmetry
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial optimization to get a good baseline
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase with occasional SA restarts
    current_r = best_r
    step_size = 0.0005
    
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    rng_sa = np.random.default_rng(456)
    
    # Temperature schedule for SA
    T_initial = 0.01
    T_min = 0.001
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        # Try multiple strategies
        success = False
        
        # Strategy 1: Direct optimization from current best
        start_pos = best_pos.copy()
        max_iter = 300 if current_r < 0.11 else 500
        new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
        actual_r = _compute_radius(new_pos)
        if actual_r >= target_r - 1e-6:
            best_pos = new_pos
            best_r = actual_r
            current_r = actual_r
            success = True
            consecutive_failures = 0
        
        # Strategy 2: If failed, try with small perturbation
        if not success and time.time() < deadline - 25.0:
            start_pos = best_pos + rng_search.normal(0, 0.005, size=(N, 3))
            start_pos = np.clip(start_pos, -0.49, 0.49)
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
        
        # Strategy 3: If still failed, try simulated annealing to escape local minimum
        if not success and time.time() < deadline - 30.0:
            # Use SA with decreasing temperature
            T = T_initial * (T_min / T_initial) ** (consecutive_failures / max_failures)
            sa_pos = _simulated_annealing_step(best_pos, current_r, T, rng_sa, n_steps=100, step_scale=0.02)
            sa_pos = np.clip(sa_pos, -0.49, 0.49)
            
            # Optimize from SA position
            new_pos, energy, opt_success = _minimize_energy(sa_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > current_r:
                # Accept improvement even if target not fully met
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
                consecutive_failures = 0
        
        if not success:
            consecutive_failures += 1
            
            # Periodic polish to maintain quality
            if time.time() < deadline - 35.0 and consecutive_failures % 3 == 0:
                polished = _polish_with_high_precision(best_pos, current_r, max_iter=500)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_pos = polished
                    best_r = polished_r
                    current_r = best_r

    # 3. Final High-Precision Polish Phase
    if time.time() < deadline - 5.0:
        # Multiple polish attempts with increasing precision
        for polish_iter in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Try to push slightly beyond current radius
            target_r = current_r + 1e-5 * (polish_iter + 1)
            max_iter = 1000 + 500 * polish_iter
            ftol = 1e-16 - polish_iter * 1e-18
            gtol = 1e-14 - polish_iter * 1e-16
            
            new_pos = _polish_with_high_precision(best_pos, target_r, max_iter=max_iter, ftol=max(ftol, 1e-18), gtol=max(gtol, 1e-16))
            actual_r = _compute_radius(new_pos)
            
            if actual_r > best_r:
                best_pos = new_pos
                best_r = actual_r
                current_r = actual_r
            else:
                # Try polishing at current radius
                new_pos = _polish_with_high_precision(best_pos, current_r, max_iter=1500, ftol=1e-18, gtol=1e-16)
                actual_r = _compute_radius(new_pos)
                if actual_r > best_r:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                else:
                    break

    # Final validation
    final_pos = np.clip(best_pos, -0.5, 0.5)
    final_r = _compute_radius(final_pos)
    
    return final_pos
# EVOLVE-BLOCK-END
