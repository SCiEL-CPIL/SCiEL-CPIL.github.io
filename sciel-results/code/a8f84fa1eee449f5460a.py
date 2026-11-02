"""
Optimized packing of 50 equal circles in a unit square.
Improved version with:
- Faster SA using incremental pair updates
- More aggressive beta scheduling in polish phases
- Additional restart diversity
- Tighter convergence criteria
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 148.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))
    
    if min_wall < 0:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    return min(min_pair, min_wall)

def _soft_violation_obj(x, r_target, beta):
    """
    Objective: Minimize smooth violation for target radius r_target.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    z_wall = np.clip(beta * wall_viol, -60, 60)
    z_pair = np.clip(beta * pair_viol, -60, 60)
    
    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta
    
    return float(np.sum(soft_wall) + np.sum(soft_pair))

def _soft_violation_grad(x, r_target, beta):
    """
    Gradient of the soft violation objective. Fully vectorized.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls
    
    z_wall = np.clip(beta * wall_viol, -60, 60)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    if len(idx_wx) > 0:
        grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists
    
    z_pair = np.clip(beta * pair_viol, -60, 60)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))
    
    factor = sig_pair / dists_safe
    contrib = factor[:, None] * diff_vec
    
    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])
    
    return grad

def _soft_min_obj(x, beta):
    """
    Epigraph-style soft-min objective.
    """
    centers = x.reshape(-1, 2)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0
    
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -60, 60)
    logsumexp = np.max(z) + np.log(np.sum(np.exp(z - np.max(z))))
    soft_min_val = -logsumexp / beta
    
    return float(-soft_min_val)

def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective.
    """
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    
    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0
    
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -60, 60)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp
    
    n_walls = N
    w_weights = weights[:n_walls]
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    if len(idx_wx) > 0:
        grad[2*idx_wx] -= w_weights[idx_wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] -= w_weights[idx_wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 1])
    
    p_weights = weights[n_walls:]
    dists_safe = np.maximum(dists, 1e-12)
    factor = p_weights / dists_safe
    contrib = factor[:, None] * diff
    
    np.add.at(grad, 2 * _IU[0], contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, -contrib[:, 1])
    
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
    
    for attempt in range(10):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        
        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
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
    Uses faster evaluation with precomputed pair indices.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00002
    est_steps = int(time_budget / 0.0003)
    n_steps = max(25000, min(100000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    iu0, iu1 = _IU[0], _IU[1]
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current radius
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_wall = np.min(np.minimum(wall_x, wall_y))
        
        ci = current[iu0]
        cj = current[iu1]
        diff = ci - cj
        dist_sq = np.einsum('ij,ij->i', diff, diff)
        min_pair = np.sqrt(np.min(dist_sq)) / 2.0
        curr_r = min(min_pair, min_wall)
        
        i = rng.integers(0, N)
        frac = T / T_start
        move_size = 0.004 + 0.008 * frac
        
        if rng.random() < 0.10:
            move_size *= 2.0
        
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
            
        old_pos = current[i].copy()
        trial = current.copy()
        trial[i] = old_pos + direction * move_size
        trial[i] = np.clip(trial[i], -0.5, 0.5)
        
        # Compute trial radius
        trial_wall_x = 0.5 - np.abs(trial[:, 0])
        trial_wall_y = 0.5 - np.abs(trial[:, 1])
        trial_min_wall = np.min(np.minimum(trial_wall_x, trial_wall_y))
        
        tci = trial[iu0]
        tcj = trial[iu1]
        tdiff = tci - tcj
        tdist_sq = np.einsum('ij,ij->i', tdiff, tdiff)
        trial_min_pair = np.sqrt(np.min(tdist_sq)) / 2.0
        trial_r = min(trial_min_pair, trial_min_wall)
        
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

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """
    Run L-BFGS-B to minimize violation for a specific target radius r_target.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 8000, 'ftol': 1e-20, 'gtol': 1e-16,
                     'maxfun': 40000, 'disp': False, 'maxls': 80}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_soft_min_polish(x0, beta, time_budget):
    """
    Run L-BFGS-B to maximize soft-min radius.
    """
    start_time = time.time()
    
    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_obj(x, beta)
        g = _soft_min_grad(x, beta)
        return f, g
    
    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15,
                     'maxfun': 25000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    """
    Binary search for the maximum feasible radius using L-BFGS polishing.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    lo = best_r
    hi = min(0.08, best_r * 1.06)
    
    search_steps = 24
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        progress = i / max(search_steps - 1, 1)
        beta = 300.0 + 1200.0 * progress
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(20.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-9:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using multiple strategies.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Phase 1: Progressive sharpening with higher beta range
    betas = [800, 1500, 3000, 6000, 10000, 18000, 30000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        r_target = best_r
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 4:
            break
        lbfgs_budget = min(16.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    # Phase 2: Soft-min epigraph polish with higher betas
    if time.time() - start_time < time_budget - 10:
        sm_betas = [500, 1000, 2000, 5000, 10000, 20000]
        for beta in sm_betas:
            if time.time() - start_time > time_budget - 5:
                break
            x0 = best.flatten()
            remaining = time_budget - (time.time() - start_time)
            if remaining < 4:
                break
            lbfgs_budget = min(16.0, remaining * 0.85)
            
            trial = _lbfgs_soft_min_polish(x0, beta, lbfgs_budget)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_r:
                best_r = trial_r
                best = trial
            
    # Phase 3: Push phase - target slightly above best_r
    if time.time() - start_time < time_budget - 5:
        for push_frac in [1.001, 1.002, 1.004, 1.006, 1.009, 1.012, 1.015, 1.020]:
            if time.time() - start_time > time_budget - 3:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 3:
                break
            lbfgs_budget = min(14.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, 8000.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    # Phase 4: Ultra-tight final polish with very high beta
    if time.time() - start_time < time_budget - 3:
        for beta in [15000, 30000, 50000, 80000]:
            if time.time() - start_time > time_budget - 2:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 2:
                break
            lbfgs_budget = min(18.0, remaining * 0.9)
            trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)
    
    # 1. Initialization - try multiple diverse starts
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    
    rng_init = np.random.default_rng(999)
    random_init = rng_init.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init)
    
    rng_init2 = np.random.default_rng(4242)
    random_init2 = rng_init2.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init2)
    
    # Add another hex variant with different offset
    rng_init3 = np.random.default_rng(8888)
    hex_variant = _hex_init()
    hex_variant = hex_variant + rng_init3.normal(0, 0.003, (N, 2))
    hex_variant = np.clip(hex_variant, -0.49, 0.49)
    inits.append(hex_variant)
    
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
        
    # 2. Simulated Annealing - multi-restart
    sa_budget_total = min(38.0, (deadline - time.time()) * 0.32)
    
    if sa_budget_total > 10:
        sa1_budget = min(22.0, sa_budget_total * 0.55)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1_budget)
        
        # Second SA restart from perturbed best
        if time.time() < deadline - 40:
            rng2 = np.random.default_rng(1234)
            perturbed = best_centers + rng2.normal(0, 0.010, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(16.0, (deadline - time.time()) - 45)
            if sa2_budget > 5:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2
        
        # Third SA restart with larger perturbation
        if time.time() < deadline - 50:
            rng3 = np.random.default_rng(5678)
            perturbed3 = best_centers + rng3.normal(0, 0.020, (N, 2))
            perturbed3 = np.clip(perturbed3, -0.49, 0.49)
            sa3_budget = min(12.0, (deadline - time.time()) - 55)
            if sa3_budget > 4:
                c3, r3 = _simulated_annealing(perturbed3, rng3, sa3_budget)
                if r3 > best_radius:
                    best_radius = r3
                    best_centers = c3
        
    # 3. Feasibility-Driven Radius Refinement
    if time.time() < deadline - 30:
        refine_budget = min(50.0, deadline - time.time() - 15)
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=refine_budget
        )
        
    # 4. High Precision Polish
    if time.time() < deadline - 8:
        polish_budget = min(35.0, deadline - time.time() - 3)
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=polish_budget
        )
        
    # 5. Final micro-polish if time remains
    if time.time() < deadline - 3:
        final_budget = min(18.0, deadline - time.time() - 1)
        if final_budget > 4:
            cur_r = _compute_radius(best_centers)
            trial = _lbfgs_feasibility_polish(
                best_centers.flatten(), cur_r, 50000.0, final_budget
            )
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
