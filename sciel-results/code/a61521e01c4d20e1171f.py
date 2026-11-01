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
    Consistent with _soft_min_objective (uses nearest neighbors only).
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances using cKDTree
    tree = cKDTree(centers)
    dists, idx = tree.query(centers, k=2)
    # dists[:, 1] is distance to nearest neighbor
    # idx[:, 1] is index of nearest neighbor
    nn_dists = dists[:, 1]
    nn_idxs = idx[:, 1]
    pair_dists = nn_dists / 2.0
    
    # We need the gradient of the soft-min of the set {walls, pair_dists}
    # Let D be the vector of all distances.
    # soft_min(D) = min(D) - tau * log(sum(exp(-(D - min(D))/tau)))
    # grad(soft_min) = - sum(w_k * grad(D_k)) where w_k are the soft-min weights
    # We are minimizing -soft_min, so we return -grad(soft_min) = sum(w_k * grad(D_k))
    
    all_dists = np.concatenate([walls, pair_dists])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    # D_wall_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # grad(D_wall_i) w.r.t x_i:
    # if 0.5 - |x_i| < 0.5 - |y_i|: -sign(x_i)
    # if 0.5 - |y_i| < 0.5 - |x_i|: -sign(y_i)
    # We want -sum(w * grad D).
    # -w * (-sign(x)) = w * sign(x)
    
    for i in range(N):
        w_wall = weights[i]
        if w_wall < 1e-10: continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            # Both are limiting
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])

    # 2. Pair contributions
    # D_pair_i = dist(i, nn(i)) / 2
    # Let j = nn(i). dist(i,j) = ||c_i - c_j||.
    # grad(D_pair_i) w.r.t c_i = (c_i - c_j) / (2 * dist(i,j))
    # We want -sum(w * grad D).
    # -w * (c_i - c_j) / (2d)
    
    for i in range(N):
        w_pair = weights[N + i]
        if w_pair < 1e-10: continue
        
        j = nn_idxs[i]
        d = nn_dists[i]
        if d < 1e-12: continue
        
        diff = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff
        grad[2*j:2*j+2] += factor * diff
        
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
    # More aggressive tau reduction and tighter tolerances
    for tau in [0.002, 0.001, 0.0005]:
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
    
    # Phase 2: L-BFGS optimization with soft-min
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        # Try with different tau values
        for tau in [0.01, 0.008, 0.005]:
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(12, deadline - time.time() - 15))
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
