"""Model-and-optimize approach for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

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
    
    # Use cKDTree for efficient nearest-neighbor distance
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding itself)
    dists, idx = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _global_repulsion_objective(x):
    """
    Objective: minimize sum of inverse square distances between all pairs and walls.
    This global potential helps escape local minima of the min-distance objective.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Add a small epsilon to avoid division by zero
    eps = 1e-6
    walls_safe = np.maximum(walls, eps)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dists_safe = np.maximum(dists, eps)
    
    # Sum of inverse squares
    pair_term = np.sum(1.0 / (dists_safe ** 2))
    wall_term = np.sum(1.0 / (walls_safe ** 2))
    
    return pair_term + wall_term

def _global_repulsion_grad(x):
    """
    Gradient of the global repulsion objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    eps = 1e-6
    walls_safe = np.maximum(walls, eps)
    
    # Wall gradient
    for i in range(N):
        w = walls_safe[i]
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += 2.0 * np.sign(centers[i, 0]) / (w ** 3)
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += 2.0 * np.sign(centers[i, 1]) / (w ** 3)
        else:
            grad[2*i] += 1.0 * np.sign(centers[i, 0]) / (w ** 3)
            grad[2*i+1] += 1.0 * np.sign(centers[i, 1]) / (w ** 3)

    # Pairwise gradient
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dists_safe = np.maximum(dists, eps)
    
    factor = -2.0 / (dists_safe ** 4)
    np.fill_diagonal(factor, 0.0)
    
    grad_pairs = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pairs
    
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    # Use a hexagonal-like arrangement for better packing
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Proposes random moves and accepts based on improvement in min radius.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.0001
    n_steps = 20000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current min radius components
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        # Find the most constrained circle(s)
        # Propose a move on a random circle, biased toward constrained ones
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        # Weight by how close to the minimum each constraint is
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.005)
        weights /= np.sum(weights)
        
        # Pick which constraint to improve
        idx = rng.choice(len(all_dists), p=weights)
        
        if idx < N:
            # Wall constraint for circle idx
            i = idx
            move_size = 0.005 + 0.01 * T / T_start
            direction = rng.integers(0, 4)
            delta = np.zeros(2)
            if direction == 0:
                delta[0] = move_size * rng.uniform(0.5, 1.0)
            elif direction == 1:
                delta[0] = -move_size * rng.uniform(0.5, 1.0)
            elif direction == 2:
                delta[1] = move_size * rng.uniform(0.5, 1.0)
            else:
                delta[1] = -move_size * rng.uniform(0.5, 1.0)
            
            trial = current.copy()
            trial[i] += delta
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Pair constraint - move one of the pair apart
            pair_idx = idx - N
            i = pair_idx
            # Find nearest neighbor
            _, nn_idx = tree.query(current[i], k=2)
            j = nn_idx[1]
            
            move_size = 0.005 + 0.01 * T / T_start
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            # Move both apart
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.7)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.7)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Evaluate
        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r
        
        # Accept if improvement, or with probability exp(delta_r / T)
        if delta_r >= 0:
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
        
        T *= cooling
    
    return best, best_r

def _lbfgs_optimize(x0, time_budget=20.0):
    """Run L-BFGS-B optimization on the global repulsion objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _global_repulsion_objective(x)
        g = _global_repulsion_grad(x)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 1000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using L-BFGS with very tight tolerances
    on the global repulsion objective to refine the solution.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Run multiple iterations with tighter tolerances
    for _ in range(3):
        if time.time() - start_time > time_budget:
            break
        
        x0 = best.flatten().copy()
        
        def obj_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            f = _global_repulsion_objective(x)
            g = _global_repulsion_grad(x)
            return f, g
        
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-18, 'gtol': 1e-14,
                         'maxfun': 3000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best = polished
        except Exception:
            pass
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(15):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use gradient descent to minimize violation for radius mid
        x = trial.flatten().copy()
        lr = 0.001
        for _ in range(80):
            if time.time() - start_time > time_budget:
                break
            c = x.reshape(-1, 2)
            
            # Compute violation
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.sum(np.maximum(0, mid - dists / 2.0))
            
            wall_x = 0.5 - np.abs(c[:, 0])
            wall_y = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wall_x, wall_y)))
            
            v = pair_viol + wall_viol
            if v < 1e-12:
                break
            
            # Gradient
            grad = np.zeros_like(x)
            iu, ju = np.triu_indices(N, k=1)
            for k in range(len(iu)):
                i, j = iu[k], ju[k]
                d = dists[i, j]
                if d < 2 * mid and d > 1e-12:
                    factor = 0.5 / d
                    diff_ij = c[i] - c[j]
                    grad[2*i:2*i+2] -= factor * diff_ij
                    grad[2*j:2*j+2] += factor * diff_ij
            for i in range(N):
                if wall_x[i] < mid:
                    grad[2*i] += np.sign(c[i, 0])
                if wall_y[i] < mid:
                    grad[2*i+1] += np.sign(c[i, 1])
            
            x -= lr * grad
            x = np.clip(x, -0.5, 0.5)
        
        trial = x.reshape(-1, 2)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing to escape local minima
    sa_budget = min(50.0, (deadline - time.time()) * 0.4)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with global repulsion
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        for _ in range(3):
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(x0, time_budget=min(12, deadline - time.time() - 15))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 15)
        )
    
    # Phase 4: Binary search with gradient polish
    if time.time() < deadline - 10:
        lo = best_radius
        hi = min(0.08, best_radius * 1.03)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(15, deadline - time.time() - 5)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
