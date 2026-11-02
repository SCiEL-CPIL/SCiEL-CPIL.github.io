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
    
    # Pairwise distances using cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(centers)
    min_pair = tree.query_pairs(r=0.5)
    if len(min_pair) == 0:
        # No pairs within 0.5, so min distance is at least 0.5/2 = 0.25
        # But we need the actual minimum
        dists = tree.query(centers, k=2)
        min_pair = np.min(dists[0][:, 1]) / 2.0
    else:
        # Find the actual minimum among close pairs
        min_d = np.inf
        for i, j in min_pair:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_d:
                min_d = d
        min_pair = min_d / 2.0
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, tau):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    x: flattened array of 2*N coordinates
    Returns (value, gradient)
    """
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Combine all constraint distances into a flat array
    iu, ju = np.triu_indices(N, k=1)
    n_pairs = len(iu)
    
    all_dists = np.empty(n_pairs + N)
    all_dists[:n_pairs] = pair_dists[iu, ju]
    all_dists[n_pairs:] = walls
    
    # Soft min calculation
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    # Avoid overflow in exp
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp)
    soft_min = min_d - tau * log_sum
    
    value = -soft_min
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(x)
    
    # 1. Wall contributions
    for i in range(N):
        w_wall = weights[n_pairs + i]
        if w_wall < 1e-12: 
            continue
        if wall_dists_x[i] < wall_dists_y[i]:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        elif wall_dists_y[i] < wall_dists_x[i]:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])
        else:
            # Both are limiting
            grad[2*i] += 0.5 * w_wall * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w_wall * np.sign(centers[i, 1])

    # 2. Pair contributions
    for k in range(n_pairs):
        w_pair = weights[k]
        if w_pair < 1e-12:
            continue
        i = iu[k]
        j = ju[k]
        d = dists[i, j]
        if d < 1e-12:
            continue
        
        diff_ij = centers[i] - centers[j]
        factor = w_pair / (2.0 * d)
        
        grad[2*i:2*i+2] -= factor * diff_ij
        grad[2*j:2*j+2] += factor * diff_ij
        
    return value, grad

def _hex_init(rng):
    """Generate a hexagonal lattice initialization."""
    # Create a hexagonal grid that fits well in the square
    # For 50 circles, use a pattern of rows with 7-8 circles each
    
    # Hexagonal packing: rows offset by half the spacing
    # Try different row configurations to get close to 50 points
    
    best_init = None
    best_score = -np.inf
    
    # Try various row configurations
    row_configs = [
        [7, 8, 7, 8, 7, 8, 7],  # 52 points
        [8, 7, 8, 7, 8, 7, 7],  # 52 points
        [7, 7, 8, 8, 8, 7, 7],  # 52 points
        [6, 8, 8, 8, 8, 8, 6],  # 52 points
        [8, 8, 7, 7, 7, 8, 8],  # 52 points
    ]
    
    for config in row_configs:
        total = sum(config)
        if total < N:
            continue
        
        # Generate hex grid with this configuration
        points = []
        spacing = 0.145  # Slightly larger than optimal to allow for boundary effects
        
        for row_idx, n_in_row in enumerate(config):
            y = (row_idx - len(config) / 2 + 0.5) * spacing * np.sqrt(3) / 2
            x_offset = (row_idx % 2) * spacing / 2
            for col in range(n_in_row):
                x = (col - n_in_row / 2 + 0.5) * spacing + x_offset
                points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N points, select the best N
        if len(points) > N:
            # Select N points that are most "interior" (furthest from boundary)
            wall_dists = np.minimum(
                np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])),
                axis=0
            ) if False else np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
            # Sort by wall distance (descending) and take top N
            idx = np.argsort(wall_dists)[::-1][:N]
            points = points[idx]
        
        # Center the grid
        cx = points[:, 0].mean()
        cy = points[:, 1].mean()
        points[:, 0] -= cx
        points[:, 1] -= cy
        
        # Scale to fit in square with margin
        max_ext = np.max(np.abs(points))
        if max_ext > 0:
            scale = 0.49 / max_ext
            points *= scale
        
        # Add small jitter
        points += rng.normal(0, 0.005, points.shape)
        points = np.clip(points, -0.49, 0.49)
        
        # Score this initialization
        score = _compute_radius(points)
        if score > best_score:
            best_score = score
            best_init = points.copy()
    
    if best_init is None:
        # Fallback: simple grid
        m = 8
        ticks = np.linspace(-0.45, 0.45, m)
        grid = np.array([(x, y) for y in ticks for x in ticks])
        idx = rng.choice(len(grid), N, replace=False)
        best_init = grid[idx] + rng.normal(0, 0.01, (N, 2))
        best_init = np.clip(best_init, -0.49, 0.49)
    
    return best_init

def _grid_init(rng):
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def _simulated_annealing(centers, time_budget, rng):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    T = T_start
    
    step_size = 0.03
    
    while time.time() - start_time < time_budget:
        # Propose a move
        idx = rng.integers(0, N)
        delta = rng.normal(0, step_size, 2)
        new_center = current[idx].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.49, 0.49)
        
        # Evaluate
        trial = current.copy()
        trial[idx] = new_center
        trial_r = _compute_radius(trial)
        
        # Accept or reject
        if trial_r > current_r:
            # Always accept improvement
            current = trial
            current_r = trial_r
            if current_r > best_r:
                best = current.copy()
                best_r = current_r
        else:
            # Accept with probability exp((new - old) / T)
            delta_r = trial_r - current_r
            if delta_r < 0:
                prob = np.exp(delta_r / T)
                if rng.random() < prob:
                    current = trial
                    current_r = trial_r
        
        # Cool down
        elapsed_frac = (time.time() - start_time) / time_budget
        T = T_start * (T_end / T_start) ** elapsed_frac
        
        # Reduce step size slightly
        step_size *= 0.999
    
    return best, best_r

def _lbfgs_optimize(x0, tau, time_budget):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f, g = _soft_min_obj_and_grad(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10,
                     'maxfun': 10000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _high_precision_polish(centers, time_budget):
    """Final high-precision polish using very tight tolerances."""
    start_time = time.time()
    current = centers.copy()
    
    # Use very small tau for precise optimization
    tau = 1e-6
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f, g = _soft_min_obj_and_grad(x, tau)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, current.flatten(), jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12,
                     'maxfun': 20000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        
        # Verify it's valid
        if _compute_radius(polished) >= _compute_radius(current) - 1e-12:
            return polished
    except Exception:
        pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(777)
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Multiple restarts with SA + L-BFGS
    n_restarts = 8
    restart_budget = min(12.0, (deadline - time.time()) / n_restarts - 5)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate different initializations
        if restart == 0:
            init = _hex_init(rng)
        elif restart == 1:
            init = _grid_init(rng)
        elif restart == 2:
            init = rng.uniform(-0.45, 0.45, (N, 2))
        else:
            # Perturb the best solution found so far
            if best_centers is not None:
                init = best_centers + rng.normal(0, 0.05, (N, 2))
                init = np.clip(init, -0.49, 0.49)
            else:
                init = _hex_init(rng)
        
        # Phase 1a: Simulated Annealing to find a good basin
        sa_budget = min(4.0, restart_budget * 0.3)
        if time.time() < deadline - 25:
            sa_result, sa_r = _simulated_annealing(init, sa_budget, rng)
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_result.copy()
            init = sa_result
        
        # Phase 1b: L-BFGS polishing with decreasing tau
        for tau in [0.01, 0.005, 0.001]:
            if time.time() > deadline - 20:
                break
            
            lbfgs_budget = min(3.0, restart_budget * 0.2)
            trial = _lbfgs_optimize(init.flatten(), tau, time_budget=lbfgs_budget)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial.copy()
            
            init = trial
    
    # Phase 2: High-precision polish on the best solution
    if best_centers is not None and time.time() < deadline - 15:
        polish_budget = min(10.0, deadline - time.time() - 5)
        polished = _high_precision_polish(best_centers, polish_budget)
        polished_r = _compute_radius(polished)
        
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished.copy()
    
    # Final Validation
    if best_centers is None or not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _grid_init(rng)
    
    return best_centers
