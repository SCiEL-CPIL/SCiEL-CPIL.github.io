"""
Packing 50 equal circles in a unit square using a force-directed / repulsion-based approach
combined with gradient optimization on a barrier objective.

Key structural change: instead of soft-min of nearest-neighbor distances, use a global
inverse-power repulsion potential that pushes all pairs apart, combined with wall barriers.
This avoids the "nearest-neighbor only" limitation and provides smoother gradients.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    return min(min_pair, min_wall)


def _hex_init():
    """Create a hexagonal close-packed initial configuration."""
    # For 50 circles, use a hexagonal lattice
    # Rows alternate between offset positions
    rows = 8
    cols = 7  # 8*7 = 56, we'll use 50

    # Hexagonal spacing
    # If we want ~50 circles in unit square [-0.5, 0.5]^2
    # Horizontal spacing: 1/cols, vertical: sqrt(3)/2 * horizontal
    # But let's just place them in a grid-like hex pattern

    centers = []
    # Use a tighter hexagonal arrangement
    # For 7 columns, 8 rows = 56 positions, take first 50
    for row in range(rows):
        for col in range(cols):
            if len(centers) >= N:
                break
            x = (col + 0.5) / cols - 0.5
            y = (row + 0.5) / rows - 0.5
            # Offset odd rows
            if row % 2 == 1:
                x += 0.5 / cols
            # Clip to stay inside
            x = np.clip(x, -0.49, 0.49)
            y = np.clip(y, -0.49, 0.49)
            centers.append([x, y])

    centers = np.array(centers[:N])
    return centers


def _grid_init():
    """Create a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))  # 8
    # Use 8x8 grid, take first 50
    centers = []
    for row in range(m):
        for col in range(m):
            if len(centers) >= N:
                break
            x = (col + 0.5) / m - 0.5
            y = (row + 0.5) / m - 0.5
            centers.append([x, y])
    centers = np.array(centers[:N])
    return centers


def _random_init(rng):
    """Random initialization inside the square."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _repulsion_objective_and_grad(x, beta=20.0, wall_beta=50.0):
    """
    Objective: sum of inverse-power repulsions between all pairs + wall barriers.
    Minimizing this pushes circles apart and away from walls.
    The minimum of this objective corresponds to a good packing.
    
    We use: f = sum_{i<j} 1/d_{ij}^beta + sum_i wall_barrier_i
    
    where wall_barrier_i = max(0, 0.5 - |x_i| - r_target)^2 + max(0, 0.5 - |y_i| - r_target)^2
    But since we don't know r_target, we use a simpler form:
    wall_barrier_i = 1/(0.5 - |x_i|)^2 + 1/(0.5 - |y_i|)^2  (with clipping to avoid inf)
    
    Actually, let's use a cleaner formulation:
    f = sum_{i<j} exp(-beta * d_{ij}) + sum_i [exp(-wall_beta * wall_dist_i)]
    
    This is smooth and the gradient is well-defined.
    """
    centers = x.reshape(-1, 2)
    
    # Pairwise repulsion: sum_{i<j} exp(-beta * d_{ij})
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only upper triangle
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dists[iu, ju]
    pair_diffs = diff[iu, ju]  # shape (M, 2) where M = N*(N-1)/2
    
    f_pair = np.sum(np.exp(-beta * pair_dists))
    
    # Wall barriers
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    # Clip to avoid division by zero or negative
    wall_x_safe = np.maximum(wall_x, 0.01)
    wall_y_safe = np.maximum(wall_y, 0.01)
    
    f_wall_x = np.sum(np.exp(-wall_beta * wall_x_safe))
    f_wall_y = np.sum(np.exp(-wall_beta * wall_y_safe))
    
    f = f_pair + f_wall_x + f_wall_y
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pair gradient: d/dx_i exp(-beta * d_{ij}) = -beta * exp(-beta * d_{ij}) * (x_i - x_j) / d_{ij}
    # For each pair (i,j): contribution to grad[i] = -beta * exp(-beta*d) * (c_i - c_j) / d
    #                 contribution to grad[j] = -beta * exp(-beta*d) * (c_j - c_i) / d = +beta * exp(-beta*d) * (c_i - c_j) / d
    
    exp_vals = np.exp(-beta * pair_dists)
    # factor for each pair
    factors = beta * exp_vals / pair_dists  # shape (M,)
    
    for k in range(len(iu)):
        i, j = iu[k], ju[k]
        d = pair_dists[k]
        if d < 1e-12:
            continue
        factor = factors[k]
        diff_ij = pair_diffs[k]  # c_i - c_j
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
    
    # Wall gradient
    # d/dx_i exp(-wall_beta * wall_x_i) where wall_x_i = 0.5 - |x_i|
    # If x_i > 0: d/dx_i (0.5 - x_i) = -1, so d/dx_i exp(-wb*(0.5-x_i)) = wb * exp(-wb*(0.5-x_i))
    # If x_i < 0: d/dx_i (0.5 + x_i) = 1, so d/dx_i exp(-wb*(0.5+x_i)) = -wb * exp(-wb*(0.5+x_i))
    # So: d/dx_i = -sign(x_i) * wb * exp(-wb * wall_x_i)
    
    for i in range(N):
        if wall_x[i] > 0.01:
            grad[2*i] += -np.sign(centers[i, 0]) * wall_beta * np.exp(-wall_beta * wall_x_safe[i])
        if wall_y[i] > 0.01:
            grad[2*i+1] += -np.sign(centers[i, 1]) * wall_beta * np.exp(-wall_beta * wall_y_safe[i])
    
    return f, grad


def _soft_min_objective_and_grad(x, tau=0.005):
    """
    Soft-min of all constraint distances (pair distances and wall distances).
    This directly optimizes the minimum radius.
    """
    centers = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu, ju = np.triu_indices(N, k=1)
    pair_dists = dists[iu, ju] / 2.0
    pair_diffs = diff[iu, ju]
    
    # Wall distances
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    # All constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    soft_min = min_d - tau * np.log(sum_exp)
    
    f = -soft_min
    
    # Gradient of -soft_min = -d(soft_min)/dx
    # d(soft_min)/dx_k = -sum_j w_j * d(D_j)/dx_k where w_j = exp(-shifted_j/tau) / sum_exp
    # So d(-soft_min)/dx_k = sum_j w_j * d(D_j)/dx_k
    
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # Wall contributions
    for i in range(N):
        w = weights[i]
        if w < 1e-12:
            continue
        if wall_x[i] < wall_y[i]:
            grad[2*i] += w * np.sign(centers[i, 0])
        elif wall_y[i] < wall_x[i]:
            grad[2*i+1] += w * np.sign(centers[i, 1])
        else:
            grad[2*i] += 0.5 * w * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w * np.sign(centers[i, 1])
    
    # Pair contributions
    for k in range(len(iu)):
        w = weights[N + k]
        if w < 1e-12:
            continue
        i, j = iu[k], ju[k]
        d = dists[i, j]  # actual distance (not halved)
        if d < 1e-12:
            continue
        # D_k = d_{ij} / 2
        # d(D_k)/d(c_i) = (c_i - c_j) / (2 * d_{ij})
        factor = w / (2.0 * d)
        diff_ij = pair_diffs[k]
        grad[2*i:2*i+2] += factor * diff_ij
        grad[2*j:2*j+2] -= factor * diff_ij
    
    return f, grad


def _lbfgs_optimize(x0, objective_type='soft_min', tau=0.005, time_budget=20.0):
    """Run L-BFGS-B optimization."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        if objective_type == 'repulsion':
            return _repulsion_objective_and_grad(x, beta=20.0, wall_beta=50.0)
        else:
            return _soft_min_objective_and_grad(x, tau)
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing with moves biased toward improving the minimum constraint.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    cooling = (T_end / T_start) ** (1.0 / 10000)
    T = T_start
    
    while time.time() - start_time < time_budget:
        # Compute current state
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        walls = np.minimum(wall_x, wall_y)
        
        min_pair = np.min(dists) / 2.0
        min_wall = np.min(walls)
        curr_r = min(min_pair, min_wall)
        
        # Find the tightest constraint
        # Combine all pair distances (halved) and wall distances
        iu, ju = np.triu_indices(N, k=1)
        pair_dists = dists[iu, ju] / 2.0
        all_dists = np.concatenate([walls, pair_dists])
        
        # Weight by proximity to minimum
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.003)
        weights /= np.sum(weights)
        
        # Pick a constraint to improve
        idx = rng.choice(len(all_dists), p=weights)
        
        move_size = 0.003 + 0.008 * (T / T_start)
        
        if idx < N:
            # Wall constraint for circle idx
            i = idx
            trial = current.copy()
            # Move away from the nearest wall
            if wall_x[i] < wall_y[i]:
                # x wall is tighter
                direction = np.sign(current[i, 0]) if current[i, 0] != 0 else rng.choice([-1, 1])
                trial[i, 0] -= direction * move_size * rng.uniform(0.5, 1.0)
            else:
                # y wall is tighter
                direction = np.sign(current[i, 1]) if current[i, 1] != 0 else rng.choice([-1, 1])
                trial[i, 1] -= direction * move_size * rng.uniform(0.5, 1.0)
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Pair constraint
            pair_idx = idx - N
            i, j = iu[pair_idx], ju[pair_idx]
            trial = current.copy()
            diff_ij = current[i] - current[j]
            d = np.linalg.norm(diff_ij)
            if d > 1e-9:
                unit = diff_ij / d
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            # Move apart
            trial[i] += unit * move_size * rng.uniform(0.3, 0.7)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.7)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Evaluate
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
        if T < T_end:
            break
    
    return best, best_r


def _multi_start_optimize(centers, rng, time_budget):
    """
    Multi-start optimization: try several perturbations and optimize each.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    n_starts = 0
    while time.time() - start_time < time_budget and n_starts < 8:
        n_starts += 1
        # Perturb the best solution
        perturbation = rng.normal(0, 0.01, size=(N, 2))
        x0 = np.clip(best + perturbation, -0.49, 0.49).flatten()
        
        # Optimize with soft-min
        trial = _lbfgs_optimize(x0, objective_type='soft_min', tau=0.005,
                                 time_budget=min(10, time_budget - (time.time() - start_time)))
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    return best, best_r


def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using progressively smaller tau values.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for tau in [0.003, 0.001, 0.0005, 0.0002]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = best.flatten().copy()
        trial = _lbfgs_optimize(x0, objective_type='soft_min', tau=tau,
                                 time_budget=min(15, time_budget - (time.time() - start_time)))
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
    
    return best, best_r


def _force_directed_relax(centers, time_budget):
    """
    Force-directed relaxation: iteratively push circles apart using inverse-square forces.
    This is a simple but effective local optimization.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    lr = 0.002
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        c = current.copy()
        forces = np.zeros_like(c)
        
        # Pair forces: repulsive
        for i in range(N):
            for j in range(i+1, N):
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-9:
                    diff = np.random.normal(size=2)
                    d = np.linalg.norm(diff)
                # Force magnitude: 1/d^2, direction: away from each other
                f_mag = 1.0 / (d * d)
                f_dir = diff / d
                forces[i] += f_mag * f_dir
                forces[j] -= f_mag * f_dir
        
        # Wall forces: push inward
        for i in range(N):
            # Left wall
            if c[i, 0] < 0:
                forces[i, 0] += 1.0 / max(0.5 - c[i, 0], 0.01)**2
            else:
                forces[i, 0] -= 1.0 / max(0.5 - c[i, 0], 0.01)**2
            # Right wall
            if c[i, 0] > 0:
                forces[i, 0] -= 1.0 / max(0.5 + c[i, 0], 0.01)**2
            else:
                forces[i, 0] += 1.0 / max(0.5 + c[i, 0], 0.01)**2
            # Bottom wall
            if c[i, 1] < 0:
                forces[i, 1] += 1.0 / max(0.5 - c[i, 1], 0.01)**2
            else:
                forces[i, 1] -= 1.0 / max(0.5 - c[i, 1], 0.01)**2
            # Top wall
            if c[i, 1] > 0:
                forces[i, 1] -= 1.0 / max(0.5 + c[i, 1], 0.01)**2
            else:
                forces[i, 1] += 1.0 / max(0.5 + c[i, 1], 0.01)**2
        
        # Update
        current = c + lr * forces
        current = np.clip(current, -0.5, 0.5)
        
        r = _compute_radius(current)
        if r > best_r:
            best_r = r
            best = current.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(42)

    # Try multiple initial configurations
    inits = [_hex_init(), _grid_init()]
    for _ in range(3):
        inits.append(_random_init(rng))

    best_centers = None
    best_radius = -1.0

    # Phase 0: Quick evaluation of initial configs
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()

    # Phase 1: Force-directed relaxation on best initial config
    fd_budget = min(15.0, (deadline - time.time()) * 0.15)
    if fd_budget > 3:
        trial, trial_r = _force_directed_relax(best_centers, fd_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 2: Simulated annealing
    sa_budget = min(30.0, (deadline - time.time()) * 0.3)
    if sa_budget > 5:
        trial, trial_r = _simulated_annealing(best_centers, rng, sa_budget)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 3: L-BFGS with repulsion objective (different landscape)
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        trial = _lbfgs_optimize(x0, objective_type='repulsion', time_budget=min(15, deadline - time.time() - 25))
        trial_r = _compute_radius(trial)
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 4: L-BFGS with soft-min objective
    if time.time() < deadline - 35:
        x0 = best_centers.flatten()
        for tau in [0.008, 0.005, 0.003]:
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(x0, objective_type='soft_min', tau=tau,
                                     time_budget=min(12, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()

    # Phase 5: Multi-start optimization
    if time.time() < deadline - 25:
        trial, trial_r = _multi_start_optimize(best_centers, rng, min(15, deadline - time.time() - 10))
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Phase 6: High-precision polish
    if time.time() < deadline - 15:
        trial, trial_r = _high_precision_polish(best_centers, min(15, deadline - time.time() - 5))
        if trial_r > best_radius:
            best_radius = trial_r
            best_centers = trial

    # Final validation
    if best_centers is None or not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _hex_init()

    return best_centers
