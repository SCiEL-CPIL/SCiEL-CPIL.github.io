"""
Optimized packing of 50 equal circles in a unit square.
Uses full-pairwise soft-min objective with L-BFGS-B and simulated annealing.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 142.0  # Safety margin before 150s kill

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
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_objective(x, tau=0.005):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    Uses ALL pairwise distances for accuracy.
    x: flattened array of 2*N coordinates
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # All pairwise distances
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    # Take minimum distance for each point
    min_pair_dists = np.min(dists, axis=1) / 2.0
    
    # Combine all constraint distances
    all_dists = np.concatenate([walls, min_pair_dists])
    
    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min

def _grad_soft_min(x, tau=0.005):
    """
    Compute gradient of -soft_min objective analytically.
    Uses all pairwise distances.
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # All pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dists = np.min(dists, axis=1) / 2.0
    
    all_dists = np.concatenate([walls, min_pair_dists])
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    for i in range(N):
        w_wall = weights[i]
        if w_wall < 1e-12:
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])
    
    # 2. Pair contributions - for each point, find its nearest neighbor
    # and compute gradient contribution
    for i in range(N):
        w_pair = weights[N + i]
        if w_pair < 1e-12:
            continue
        
        # Find nearest neighbor of point i
        dists_to_i = dists[i].copy()
        dists_to_i[i] = np.inf
        j = np.argmin(dists_to_i)
        d = dists_to_i[j]
        
        if d < 1e-12:
            continue
        
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
    
    return grad

def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0
    
    rng = np.random.default_rng(42)
    
    for attempt in range(8):
        # Try different row/column configurations
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        # Calculate spacing to fit
        s_width = 1.0 / (cols + 0.3)
        s_height = 1.0 / (rows * 0.866 + 0.3)
        s = min(s_width, s_height)
        
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()
    
    return best_grid if best_grid is not None else _initial_grid()

def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    Optimized: build KDTree once per step, not per evaluation.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    n_steps = 15000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Build tree once per step
        tree = cKDTree(current)
        dists, idx = tree.query(current, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)
        
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.004)
        weights /= np.sum(weights)
        
        idx_sel = rng.choice(len(all_dists), p=weights)
        
        if idx_sel < N:
            i = idx_sel
            move_size = 0.003 + 0.012 * T / T_start
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
            pair_idx = idx_sel - N
            i = pair_idx
            j = idx[i, 1]
            
            move_size = 0.003 + 0.012 * T / T_start
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.7)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.7)
            trial = np.clip(trial, -0.5, 0.5)
        
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
    
    return best, best_r

def _lbfgs_optimize(x0, tau=0.005, time_budget=15.0):
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
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using L-BFGS with very tight tolerances and decreasing tau.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    for tau in [0.002, 0.001, 0.0005, 0.0002]:
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
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15,
                         'maxfun': 5000, 'disp': False}
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

def _violation_objective(x, target_r):
    """
    Smooth violation surrogate: sum of squared violations for radius target_r.
    Uses all pairwise distances.
    """
    centers = x.reshape(-1, 2)
    
    # Wall violations
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    wall_viol = np.maximum(0, target_r - wall_dists)
    
    # Pair violations using all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dists = np.min(dists, axis=1) / 2.0
    pair_viol = np.maximum(0, target_r - min_pair_dists)
    
    return np.sum(wall_viol**2) + np.sum(pair_viol**2)

def _violation_grad(x, target_r):
    """Gradient of the violation objective using all pairwise distances."""
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Wall gradient
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    
    for i in range(N):
        v = target_r - wall_dists[i]
        if v > 0:
            if wall_x[i] <= wall_y[i]:
                grad[2*i] += 2 * v * np.sign(centers[i, 0])
            else:
                grad[2*i+1] += 2 * v * np.sign(centers[i, 1])
    
    # Pair gradient using all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(N):
        # Find nearest neighbor
        dists_i = dists[i].copy()
        dists_i[i] = np.inf
        j = np.argmin(dists_i)
        d = dists_i[j]
        
        if d < 1e-12:
            continue
        v = target_r - d / 2.0
        if v > 0:
            diff_ij = centers[i] - centers[j]
            factor = -v / d
            grad[2*i:2*i+2] += factor * diff_ij
            grad[2*j:2*j+2] -= factor * diff_ij
    
    return grad

def _projected_gradient_feasibility(centers, target_r, max_iter=150, time_budget=8.0):
    """
    Projected gradient descent to find a feasible configuration for target_r.
    Returns (success, centers).
    """
    start_time = time.time()
    x = centers.flatten().copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Check if already feasible
        c = x.reshape(-1, 2)
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.minimum(wall_x, wall_y)
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0
        
        if np.min(wall_dists) >= target_r - 1e-10 and np.min(pair_dists) >= target_r - 1e-10:
            return True, c
        
        # Compute gradient
        g = _violation_grad(x, target_r)
        
        # Adaptive step size
        obj = _violation_objective(x, target_r)
        lr = 0.01
        for _ in range(12):
            x_new = x - lr * g
            x_new = np.clip(x_new, -0.5, 0.5)
            obj_new = _violation_objective(x_new, target_r)
            if obj_new < obj:
                x = x_new
                break
            lr *= 0.5
        
        if lr < 1e-9:
            break
    
    # Final check
    c = x.reshape(-1, 2)
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    tree = cKDTree(c)
    dists, _ = tree.query(c, k=2)
    pair_dists = dists[:, 1] / 2.0
    
    success = np.min(wall_dists) >= target_r - 1e-9 and np.min(pair_dists) >= target_r - 1e-9
    return success, c

def _binary_search_radius(centers, lo, hi, time_budget):
    """Binary search for maximum feasible radius with projected gradient polishing."""
    start_time = time.time()
    best = centers.copy()
    best_r = lo
    
    for _ in range(10):
        if time.time() - start_time > time_budget:
            break
        
        mid = (lo + hi) / 2
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        success, trial = _projected_gradient_feasibility(
            best, mid, max_iter=120, 
            time_budget=min(6.0, remaining)
        )
        
        trial_r = _compute_radius(trial)
        
        if success or trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try multiple initializations
    rng = np.random.default_rng(777)
    
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    
    best_centers = None
    best_radius = 0
    
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)
    
    # Phase 1: Simulated Annealing
    sa_budget = min(35.0, (deadline - time.time()) * 0.30)
    if sa_budget > 5:
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa_budget)
    
    # Phase 2: L-BFGS optimization with soft-min, multiple tau values
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        for tau in [0.01, 0.008, 0.005, 0.003]:
            if time.time() > deadline - 30:
                break
            remaining = deadline - time.time() - 10
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(12, remaining))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(15, deadline - time.time() - 10)
        )
    
    # Phase 4: Binary search with projected gradient feasibility
    if time.time() < deadline - 10:
        lo = best_radius
        hi = min(0.075, best_radius * 1.04)
        best_centers, best_radius = _binary_search_radius(
            best_centers, lo, hi, time_budget=min(18, deadline - time.time() - 5)
        )
    
    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    
    return best_centers
