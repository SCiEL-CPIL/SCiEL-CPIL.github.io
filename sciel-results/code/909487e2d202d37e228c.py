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

def _soft_min_objective(x, tau=0.008):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.008):
    """
    Compute gradient of -soft_min objective analytically using cKDTree for speed.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    all_dists = np.concatenate([walls, pair_dists])
    
    # Soft-min weights: w_k = exp(-d_k/tau) / sum(exp(-d_j/tau))
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] <= wall_dists_x[i] + 1e-15:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] <= wall_dists_y[i] + 1e-15:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions (each point's nearest neighbor)
    for i in range(N):
        w_pair = weights[N + i]
        j = idx[i, 1]
        d = dists[i, 1]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
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
    Uses faster evaluation and adaptive move sizes.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    # Temperature schedule
    T_start = 0.015
    T_end = 0.00005
    n_steps = 30000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    # Precompute initial tree
    tree = cKDTree(current)
    dists, idx_nn = tree.query(current, k=2)
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    min_pair = np.min(dists[:, 1]) / 2.0
    min_wall = np.min(np.minimum(wall_x, wall_y))
    curr_r = min(min_pair, min_wall)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Weighted selection of constraint to improve
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx = rng.choice(len(all_dists), p=weights)
        
        move_size = 0.003 + 0.008 * (T / T_start)
        
        if idx < N:
            # Wall constraint for circle idx
            i = idx
            # Move away from the limiting wall
            if wall_x[i] <= wall_y[i]:
                delta = np.array([np.sign(current[i, 0]) if current[i, 0] != 0 else rng.choice([-1, 1]), 0.0])
            else:
                delta = np.array([0.0, np.sign(current[i, 1]) if current[i, 1] != 0 else rng.choice([-1, 1])])
            # Add some randomness
            delta += rng.normal(0, 0.002, 2)
            trial = current.copy()
            trial[i] += delta * move_size
            trial = np.clip(trial, -0.5, 0.5)
        else:
            # Pair constraint - move apart from nearest neighbor
            i = idx - N
            j = idx_nn[i, 1]
            
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            # Move both apart with some randomness
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.8)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.8)
            # Add small random perturbation
            trial[i] += rng.normal(0, 0.001, 2)
            trial[j] += rng.normal(0, 0.001, 2)
            trial = np.clip(trial, -0.5, 0.5)
        
        # Fast evaluation
        trial_tree = cKDTree(trial)
        trial_dists, _ = trial_tree.query(trial, k=2)
        trial_min_pair = np.min(trial_dists[:, 1]) / 2.0
        trial_wall_x = 0.5 - np.abs(trial[:, 0])
        trial_wall_y = 0.5 - np.abs(trial[:, 1])
        trial_min_wall = np.min(np.minimum(trial_wall_x, trial_wall_y))
        trial_r = min(trial_min_pair, trial_min_wall)
        
        delta_r = trial_r - curr_r
        
        # Accept if improvement, or with probability exp(delta_r / T)
        if delta_r >= 0:
            current = trial
            curr_r = trial_r
            # Update cached values
            tree = trial_tree
            dists = trial_dists
            idx_nn = trial_tree.query(trial, k=2)[1]
            wall_x = trial_wall_x
            wall_y = trial_wall_y
            min_pair = trial_min_pair
            min_wall = trial_min_wall
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
                curr_r = trial_r
                tree = trial_tree
                dists = trial_dists
                idx_nn = trial_tree.query(trial, k=2)[1]
                wall_x = trial_wall_x
                wall_y = trial_wall_y
                min_pair = trial_min_pair
                min_wall = trial_min_wall
        
        T *= cooling
    
    return best, best_r

def _lbfgs_optimize(x0, tau=0.008, time_budget=20.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
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
    on a smaller soft-min tau to refine the solution.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Use progressively smaller tau for tighter optimization
    for tau in [0.004, 0.002, 0.001, 0.0005]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = best.flatten().copy()
        
        def obj_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            f = _soft_min_objective(x, tau)
            g = _grad_soft_min(x, tau)
            return f, g
        
        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-13,
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

def _exact_local_polish(centers, time_budget):
    """
    Direct local optimization: identify the bottleneck constraints and
    perform a focused coordinate ascent to maximize the minimum radius.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for iteration in range(50):
        if time.time() - start_time > time_budget:
            break
        
        c = best.copy()
        
        # Compute all constraint distances
        tree = cKDTree(c)
        dists, idx_nn = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.minimum(wall_x, wall_y)
        
        all_dists = np.concatenate([wall_dists, pair_dists])
        min_r = np.min(all_dists)
        
        # Find all constraints within a small tolerance of the minimum
        tol = 1e-6
        active_wall = np.where(wall_dists < min_r + tol)[0]
        active_pair = np.where(pair_dists < min_r + tol)[0]
        
        if len(active_wall) == 0 and len(active_pair) == 0:
            break
        
        # Try to move circles to increase the minimum
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Compute the limiting distance for circle i
            my_wall = wall_dists[i]
            my_pair = pair_dists[i]
            my_min = min(my_wall, my_pair)
            
            if my_min > min_r + tol:
                continue
            
            # Try small moves in random directions
            for _ in range(20):
                if time.time() - start_time > time_budget:
                    break
                
                direction = rng_global.normal(size=2)
                direction /= np.linalg.norm(direction) + 1e-15
                
                for step in [0.001, 0.0005, 0.0002, 0.0001]:
                    trial = c.copy()
                    trial[i] += direction * step
                    trial = np.clip(trial, -0.5, 0.5)
                    
                    trial_r = _compute_radius(trial)
                    if trial_r > best_r + 1e-12:
                        best = trial
                        best_r = trial_r
                        c = trial
                        improved = True
                        break
                    elif trial_r >= min_r - 1e-9:
                        # Accept if it doesn't worsen significantly
                        c = trial
        
        if not improved:
            break
    
    return best, best_r

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(12):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        trial = best.copy()
        
        # Use L-BFGS to minimize violation for radius mid
        x = trial.flatten().copy()
        
        def violation_obj(x):
            c = x.reshape(-1, 2)
            tree = cKDTree(c)
            d, _ = tree.query(c, k=2)
            pair_viol = np.sum(np.maximum(0, mid - d[:, 1] / 2.0))
            wx = 0.5 - np.abs(c[:, 0])
            wy = 0.5 - np.abs(c[:, 1])
            wall_viol = np.sum(np.maximum(0, mid - np.minimum(wx, wy)))
            return pair_viol + wall_viol
        
        try:
            result = opt.minimize(
                violation_obj, x, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-14, 'maxfun': 500, 'disp': False}
            )
            trial = result.x.reshape(-1, 2)
            trial = np.clip(trial, -0.5, 0.5)
        except Exception:
            pass
        
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            if trial_r > best_r:
                best_r = trial_r
                best = trial
        else:
            hi = mid
    
    return best, best_r

# Global RNG for use in _exact_local_polish
rng_global = np.random.default_rng(42)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(777)
    
    # Phase 1: Simulated Annealing to escape local minima
    sa_budget = min(55.0, (deadline - time.time()) * 0.45)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 45:
        x0 = best_centers.flatten()
        # Try with different tau values
        for tau in [0.01, 0.007, 0.004]:
            if time.time() > deadline - 30:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(12, deadline - time.time() - 20))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 30:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(25, deadline - time.time() - 15)
        )
    
    # Phase 4: Exact local polish
    if time.time() < deadline - 15:
        best_centers, best_radius = _exact_local_polish(
            best_centers, time_budget=min(10, deadline - time.time() - 8)
        )
    
    # Phase 5: Binary search with gradient polish
    if time.time() < deadline - 8:
        lo = best_radius
        hi = min(0.08, best_radius * 1.02)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(8, deadline - time.time() - 3)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
