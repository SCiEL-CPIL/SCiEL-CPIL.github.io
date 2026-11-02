"""
Optimized packing of 50 equal circles in a unit square.
Combines Simulated Annealing for global search with high-precision L-BFGS polishing
using a full-pairwise smooth violation objective, plus a soft-min epigraph approach
for final polishing.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

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
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)
    
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
    
    z_wall = np.clip(beta * wall_viol, -50, 50)
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
    
    z_pair = np.clip(beta * pair_viol, -50, 50)
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
    Epigraph-style soft-min objective: minimize the soft-min of all constraints.
    The constraints are: pairwise distances >= 2r and wall distances >= r.
    We use a fixed r_target that we try to maximize via outer loop.
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
    
    # All constraints values (we want them all >= 0, i.e. r <= min)
    # Soft-min of all values: -logsumexp(-beta * values) / beta
    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    logsumexp = np.max(z) + np.log(np.sum(np.exp(z - np.max(z))))
    soft_min_val = -logsumexp / beta
    
    return float(-soft_min_val)  # minimize negative soft-min

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
    z = np.clip(z, -50, 50)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp  # softmax weights
    
    # Gradient of -soft_min = -(-1/beta) * d(logsumexp)/dx = (1/beta) * sum_k weights_k * d(-val_k)/dx
    # = -(1/beta) * sum_k weights_k * d(val_k)/dx
    # Wait: soft_min = -logsumexp(-beta*vals)/beta
    # d(soft_min)/dx = -1/beta * d(logsumexp(-beta*vals))/dx
    # = -1/beta * sum_k exp(-beta*val_k)/sum_j exp(-beta*val_j) * (-beta) * d(val_k)/dx
    # = sum_k weights_k * d(val_k)/dx
    # We want to minimize -soft_min, so gradient is -sum_k weights_k * d(val_k)/dx
    
    # Wall gradients
    n_walls = N
    w_weights = weights[:n_walls]
    
    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)
    
    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]
    
    # d(wall_i)/d(x_i) = -sign(x_i) if x is limiting, -sign(y_i) if y is limiting
    if len(idx_wx) > 0:
        grad[2*idx_wx] -= w_weights[idx_wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] -= w_weights[idx_wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 1])
    
    # Pair gradients
    p_weights = weights[n_walls:]
    dists_safe = np.maximum(dists, 1e-12)
    # d(half_dist_ij)/d(c_i) = -(c_i - c_j) / dist_ij
    # d(half_dist_ij)/d(c_j) = (c_i - c_j) / dist_ij
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
    
    for attempt in range(8):
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
    Uses incremental evaluation: only recompute distances involving the moved point.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)
    
    T_start = 0.015
    T_end = 0.00005
    est_steps = int(time_budget / 0.00025)
    n_steps = max(20000, min(100000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start
    
    iu = _IU
    # Precompute which pairs involve each point for fast lookup
    # For point i, pairs are those where _IU[0]==i or _IU[1]==i
    point_pair_idx = [[] for _ in range(N)]
    for k in range(_NPAIRS):
        a, b = _IU[0][k], _IU[1][k]
        point_pair_idx[a].append(k)
        point_pair_idx[b].append(k)
    
    # Compute initial full distance array
    ci_all = current[iu[0]]
    cj_all = current[iu[1]]
    diff_all = ci_all - cj_all
    dist_sq_all = np.einsum('ij,ij->i', diff_all, diff_all)
    
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    min_pair = np.sqrt(np.min(dist_sq_all)) / 2.0
    curr_r = min(min_pair, min_wall)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        i = rng.integers(0, N)
        move_size = 0.002 + 0.008 * (T / T_start)
        
        if rng.random() < 0.05:
            move_size *= 3.0
        
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
        
        old_pos = current[i].copy()
        new_pos = old_pos + direction * move_size
        # Clip to stay in bounds
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Compute trial radius using only affected pairs
        # Wall for point i
        twx = 0.5 - abs(new_pos[0])
        twy = 0.5 - abs(new_pos[1])
        trial_wall_i = min(twx, twy)
        
        # Pairs involving point i
        affected = point_pair_idx[i]
        if len(affected) > 0:
            # Get the other endpoints
            other_pts = []
            for k in affected:
                if _IU[0][k] == i:
                    other_pts.append(_IU[1][k])
                else:
                    other_pts.append(_IU[0][k])
            
            other_pos = current[other_pts]
            diffs = new_pos[None, :] - other_pos
            new_dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            min_affected_pair = np.min(new_dists) / 2.0
        else:
            min_affected_pair = np.inf
        
        # Trial min pair: min of unaffected pairs and affected pairs
        if len(affected) < _NPAIRS:
            # Unaffected pairs keep their old distances
            mask = np.ones(_NPAIRS, dtype=bool)
            mask[affected] = False
            if np.any(mask):
                min_unaffected = np.sqrt(np.min(dist_sq_all[mask])) / 2.0
            else:
                min_unaffected = np.inf
            trial_min_pair = min(min_unaffected, min_affected_pair)
        else:
            trial_min_pair = min_affected_pair
        
        trial_min_wall = min(trial_wall_i, np.min(np.minimum(wall_x, wall_y)))
        # Actually we need to be more careful: min_wall is the global min
        # If point i's wall was the min, we need to recompute
        # For speed, use: trial_min_wall = min(trial_wall_i, min of all other walls)
        # But that's expensive. Approximation: if trial_wall_i >= old wall_i, 
        # then trial_min_wall = min_wall (since other walls unchanged)
        # If trial_wall_i < old wall_i, trial_min_wall = min(trial_wall_i, min of others)
        old_wall_i = min(wall_x[i], wall_y[i])
        if trial_wall_i >= old_wall_i:
            trial_min_wall = min_wall
        else:
            # Need to find min of all walls except possibly i
            # Quick check: if min_wall was from point i, we need second min
            # For speed, just compute min of all trial walls (N=50, cheap)
            trial_walls = np.minimum(wall_x, wall_y).copy()
            trial_walls[i] = trial_wall_i
            trial_min_wall = np.min(trial_walls)
        
        trial_r = min(trial_min_pair, trial_min_wall)
        delta_r = trial_r - curr_r
        
        if delta_r >= 0 or (rng.random() < np.exp(delta_r / max(T, 1e-10))):
            current[i] = new_pos
            # Update state
            wall_x[i] = 0.5 - abs(new_pos[0])
            wall_y[i] = 0.5 - abs(new_pos[1])
            
            if len(affected) > 0:
                other_pts = []
                for k in affected:
                    if _IU[0][k] == i:
                        other_pts.append(_IU[1][k])
                    else:
                        other_pts.append(_IU[0][k])
                other_pos = current[other_pts]
                diffs = new_pos[None, :] - other_pos
                new_dists = np.sum(diffs ** 2, axis=1)
                dist_sq_all[affected] = new_dists
            
            min_wall = np.min(np.minimum(wall_x, wall_y))
            min_pair = np.sqrt(np.min(dist_sq_all)) / 2.0
            curr_r = min(min_pair, min_wall)
            
            if curr_r > best_r:
                best_r = curr_r
                best = current.copy()
        
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
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15,
                     'maxfun': 30000, 'maxls': 50, 'disp': False}
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
                     'maxfun': 30000, 'maxls': 50, 'disp': False}
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
    hi = min(0.08, best_r * 1.05)
    
    search_steps = 22
    for i in range(search_steps):
        if time.time() - start_time > time_budget:
            break
            
        mid = (lo + hi) / 2
        # Use progressively higher beta as we converge
        progress = i / max(search_steps - 1, 1)
        beta = 200.0 + 800.0 * progress
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(12.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, mid, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r >= mid - 1e-8:
            lo = mid
            best = trial
            best_r = trial_r
        else:
            hi = mid
            
    return best, best_r

def _high_precision_polish(centers, time_budget):
    """
    Final high-precision polish using multiple strategies:
    1. Progressive sharpening with feasibility objective
    2. Soft-min epigraph polish
    3. Push phase targeting slightly above current best
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Phase 1: Progressive sharpening of violation objective
    betas = [500, 1000, 2000, 4000, 8000, 12000, 16000]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        r_target = best_r
        x0 = best.flatten()
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(10.0, remaining * 0.85)
        
        trial = _lbfgs_feasibility_polish(x0, r_target, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)
        
        if trial_r > best_r:
            best_r = trial_r
            best = trial
            
    # Phase 2: Soft-min epigraph polish
    if time.time() - start_time < time_budget - 10:
        sm_betas = [200, 500, 1000, 2000, 4000, 8000]
        for beta in sm_betas:
            if time.time() - start_time > time_budget - 5:
                break
            x0 = best.flatten()
            remaining = time_budget - (time.time() - start_time)
            if remaining < 5:
                break
            lbfgs_budget = min(10.0, remaining * 0.85)
            
            trial = _lbfgs_soft_min_polish(x0, beta, lbfgs_budget)
            trial_r = _compute_radius(trial)
            
            if trial_r > best_r:
                best_r = trial_r
                best = trial
            
    # Phase 3: Push phase - target slightly above best_r
    if time.time() - start_time < time_budget - 5:
        for push_frac in [1.001, 1.002, 1.004, 1.006, 1.008, 1.01, 1.012, 1.015, 1.018, 1.02]:
            if time.time() - start_time > time_budget - 3:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 3:
                break
            lbfgs_budget = min(8.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, 4000.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)
    
    # 1. Initialization
    inits = [
        _initial_grid(),
        _hex_init(),
    ]
    
    rng_init = np.random.default_rng(999)
    random_init = rng_init.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init)
    
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
    sa_budget_total = min(55.0, (deadline - time.time()) * 0.45)
    
    if sa_budget_total > 10:
        sa1_budget = min(28.0, sa_budget_total * 0.5)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1_budget)
        
        if time.time() < deadline - 30:
            rng2 = np.random.default_rng(1234)
            perturbed = best_centers + rng2.normal(0, 0.01, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(27.0, (deadline - time.time()) - 35)
            if sa2_budget > 5:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2
        
        # Third SA restart from a different perturbation
        if time.time() < deadline - 25:
            rng3 = np.random.default_rng(5678)
            perturbed3 = best_centers + rng3.normal(0, 0.015, (N, 2))
            perturbed3 = np.clip(perturbed3, -0.49, 0.49)
            sa3_budget = min(20.0, (deadline - time.time()) - 30)
            if sa3_budget > 5:
                c3, r3 = _simulated_annealing(perturbed3, rng3, sa3_budget)
                if r3 > best_radius:
                    best_radius = r3
                    best_centers = c3
        
    # 3. Feasibility-Driven Radius Refinement
    if time.time() < deadline - 30:
        best_centers, best_radius = _refine_radius(
            best_centers, time_budget=min(45.0, deadline - time.time() - 15)
        )
        
    # 4. High Precision Polish
    if time.time() < deadline - 8:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(30.0, deadline - time.time() - 3)
        )
        
    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
        
    return best_centers
