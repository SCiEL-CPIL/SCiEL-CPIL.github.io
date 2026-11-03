"""
Optimization of 100 equal spheres in a unit cube.
Uses smooth soft-min energy with L-BFGS-B and a final polish pass.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

TIME_LIMIT = 115.0  # safety margin before 120s kill


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


def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth surrogate for min(min_pair_dist/2, min_wall_dist).
    """
    centers = centers.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Pairwise distances (upper triangle only)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    np.fill_diagonal(dists, np.inf)
    
    # Use upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Combine all constraints: we want min of {pair_dists/2, wall_dists}
    # Use soft-min via log-sum-exp
    all_constraints = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return soft_min


def _energy_and_gradient(x, R, beta=50.0):
    """
    Calculates smooth energy and gradient for a target radius R.
    Uses vectorized soft-min based energy to avoid nonsmoothness.
    Energy = max(0, R - soft_min_radius)^2 + penalty for being outside cube
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Compute all constraint values
    # Wall distances
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    # Pairwise distances - vectorized
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_dists_safe = np.maximum(pair_dists, 1e-15)
    
    # All constraints: pair_dists/2 and wall_dists
    all_constraints = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    soft_min_val = min_val - np.log(sum_exp) / beta
    
    # Energy
    if soft_min_val < R:
        violation = R - soft_min_val
        energy = violation ** 2
    else:
        energy = 0.0
    
    # Gradient computation - vectorized
    weights = exp_vals / sum_exp
    
    # Gradient of soft_min with respect to centers
    grad_soft_min = np.zeros_like(centers)
    
    # Pairwise contributions - vectorized using np.add.at
    n_pairs = len(i_idx)
    pair_weights = weights[:n_pairs]
    
    # Compute unit vectors for all pairs
    pair_vecs = (centers[i_idx] - centers[j_idx]) / pair_dists_safe[:, np.newaxis]
    
    # Each pair contributes 0.5 * w * vec to center i and -0.5 * w * vec to center j
    contrib_i = 0.5 * pair_weights[:, np.newaxis] * pair_vecs
    contrib_j = -0.5 * pair_weights[:, np.newaxis] * pair_vecs
    
    np.add.at(grad_soft_min, i_idx, contrib_i)
    np.add.at(grad_soft_min, j_idx, contrib_j)
    
    # Wall contributions - vectorized
    wall_weights = weights[n_pairs:]
    # Find which dimension has max abs for each center
    max_dim = np.argmax(abs_centers, axis=1)
    signs = np.sign(centers[np.arange(n), max_dim])
    
    # d(wall_dist)/d(centers[i,k]) = -sign(centers[i,k]) for the k with max abs
    for i in range(n):
        grad_soft_min[i, max_dim[i]] += wall_weights[i] * (-signs[i])
    
    # Energy gradient
    if soft_min_val < R:
        factor = -2.0 * (R - soft_min_val)
        grad = factor * grad_soft_min
    else:
        grad = 0.01 * grad_soft_min
    
    # Add hard boundary penalty
    for dim in range(3):
        below = centers[:, dim] < -0.5
        above = centers[:, dim] > 0.5
        if np.any(below):
            penalty = (centers[below, dim] + 0.5) ** 2
            energy += np.sum(penalty)
            grad[below, dim] += 2.0 * (centers[below, dim] + 0.5)
        if np.any(above):
            penalty = (centers[above, dim] - 0.5) ** 2
            energy += np.sum(penalty)
            grad[above, dim] += 2.0 * (centers[above, dim] - 0.5)
    
    return energy, grad.flatten()


def _simulated_annealing(centers, time_budget=30.0, n_iter_max=50000):
    """
    Simulated annealing to escape local minima.
    Tries small random perturbations and accepts based on temperature schedule.
    """
    start_time = time.time()
    rng = np.random.RandomState(12345)
    
    current_centers = centers.copy()
    current_radius = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_radius = current_radius
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    n_iter = min(n_iter_max, int(time_budget * 500))  # estimate iterations based on time
    
    for it in range(n_iter):
        if time.time() - start_time > time_budget - 2.0:
            break
        
        # Temperature decay (geometric)
        T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        
        # Generate perturbation
        # Pick 1-3 random spheres to move
        n_move = rng.randint(1, 4)
        move_idx = rng.choice(N, size=n_move, replace=False)
        
        # Small random displacement
        scale = T * 0.5
        displacement = rng.normal(0, scale, size=(n_move, 3))
        
        new_centers = current_centers.copy()
        new_centers[move_idx] += displacement
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_radius = _compute_radius(new_centers)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
            current_centers = new_centers
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = current_centers.copy()
    
    return best_centers


def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    start_time = time.time()
    current_radius = _compute_radius(centers)
    
    # Use a slightly smaller target to ensure we're in the feasible region
    target_R = current_radius * 0.999
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=100.0)
        return val, g

    bounds = [(-0.49999, 0.49999)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.49999, 0.49999)
        polished_radius = _compute_radius(polished)
        
        if polished_radius >= current_radius and time.time() - start_time < time_budget:
            return polished
    except Exception:
        pass
    
    return centers


def _optimize_for_radius(init_centers, target_R, max_iter=200, beta=50.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=beta)
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
        
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_radius = 0.0
    best_centers = None
    
    # Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/ny, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # 5x4x5 = 100
    grid_centers = make_grid(5, 4, 5)
    
    # Random Initializations
    rng = np.random.RandomState(42)
    inits = [grid_centers.copy()]
    
    for i in range(7):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
    
    for init in inits:
        if time.time() > deadline - 25.0:
            break
            
        # Calculate initial radius
        r_est = _compute_radius(init)
        
        # Start with a slightly larger target
        target_R = max(r_est * 1.02, 0.06)
        
        current_centers = init.copy()
        current_radius = r_est
        
        # Iterative optimization with increasing radius
        stuck_count = 0
        for step in range(50):
            if time.time() > deadline - 20.0:
                break
            
            # Optimize for current target_R
            opt_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=150, beta=50.0)
            opt_radius = _compute_radius(opt_centers)
            
            if opt_radius > current_radius + 1e-10:
                current_centers = opt_centers
                current_radius = opt_radius
                stuck_count = 0
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_centers.copy()
                
                # Increase target slightly
                target_R = current_radius * 1.003
            else:
                stuck_count += 1
                
                if stuck_count >= 3 and time.time() < deadline - 30.0:
                    # Try simulated annealing to escape local minimum
                    sa_time = min(15.0, deadline - 25.0 - time.time())
                    if sa_time > 5.0:
                        sa_centers = _simulated_annealing(current_centers, time_budget=sa_time)
                        sa_radius = _compute_radius(sa_centers)
                        
                        if sa_radius > current_radius:
                            current_centers = sa_centers
                            current_radius = sa_radius
                            target_R = current_radius * 1.002
                            stuck_count = 0
                        else:
                            break
                elif energy < 1e-8:
                    # Very small energy, we're at a local optimum for this R
                    # Try a much smaller increase
                    target_R = current_radius * 1.001
                else:
                    # Not converged, try to heal
                    heal_R = target_R * 0.995
                    heal_centers, heal_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50, beta=50.0)
                    heal_radius = _compute_radius(heal_centers)
                    
                    if heal_radius > current_radius:
                        current_centers = heal_centers
                        current_radius = heal_radius
                        target_R = current_radius * 1.002
                    else:
                        # Stuck, move to next init
                        break
        
        # Update best
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final polish pass
    if best_centers is not None and time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        if remaining > 10.0:
            # Try SA one more time if we have enough time
            sa_time = min(15.0, remaining - 10.0)
            if sa_time > 5.0:
                sa_centers = _simulated_annealing(best_centers, time_budget=sa_time)
                sa_radius = _compute_radius(sa_centers)
                if sa_radius > best_radius:
                    best_radius = sa_radius
                    best_centers = sa_centers
            
            # Final polish
            polished = _polish_centers(best_centers, max_iter=300, time_budget=min(10.0, deadline - 5.0 - time.time()))
            polished_radius = _compute_radius(polished)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished
    
    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
        
    return best_centers
# EVOLVE-BLOCK-END
