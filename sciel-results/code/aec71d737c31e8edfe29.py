"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 150s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best

def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def _soft_min_obj_and_grad(x, beta=80.0):
    """
    Vectorized soft-min of all pair distances and wall distances, divided by 2.
    Objective: maximize soft_min / 2 (so we minimize negative of it).
    Returns (f, g) where f = -soft_min/2 and g = gradient of f w.r.t. x.
    
    x: shape (N*2,) flattened centers
    """
    c = x.reshape(N, 2)
    
    # Pair distances (upper triangle)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = np.sqrt(dists_sq[mask])
    
    # Wall distances: min(0.5-|x|, 0.5-|y|) for each circle
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    # All distances
    all_dists = np.concatenate([dists, wall_dists])
    
    # Soft min
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    weights = exp_vals / sum_exp
    
    # Objective: minimize -soft_min / 2
    f = -soft_min / 2.0
    
    # Gradient: d(soft_min)/d(c_i) = -sum_k w_k * d(dist_k)/d(c_i)
    # So d(f)/d(c_i) = -0.5 * d(soft_min)/d(c_i) = 0.5 * sum_k w_k * d(dist_k)/d(c_i)
    grad_c = np.zeros((N, 2))
    
    # Pair distance gradients (vectorized)
    pair_weights = weights[:N * (N - 1) // 2]
    # Reconstruct pair indices
    i_idx, j_idx = np.where(mask)
    # i_idx and j_idx are for all pairs where mask is True (upper triangle)
    # But we need to ensure i < j ordering matches our dists extraction
    # np.where(mask) gives row, col where mask is True. For upper triangle, row < col.
    # Our dists was extracted as dists_sq[mask] which follows row-major order.
    # Let's verify: for N=50, mask has True at (i,j) where i<j.
    # np.where(mask) returns arrays of i's and j's in row-major order.
    
    # d(dist_ij)/d(c_i) = (c_i - c_j) / dist_ij
    # d(dist_ij)/d(c_j) = (c_j - c_i) / dist_ij
    pair_diff = c[i_idx] - c[j_idx]  # shape (num_pairs, 2)
    pair_dists = dists  # shape (num_pairs,)
    safe_dists = np.maximum(pair_dists, 1e-12)
    pair_dirs = pair_diff / safe_dists[:, np.newaxis]  # (num_pairs, 2)
    
    # grad_c[i] += w * pair_dir for each pair (i,j)
    # grad_c[j] -= w * pair_dir for each pair (i,j)
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w > 1e-15:
            grad_c[i_idx[k]] += w * pair_dirs[k]
            grad_c[j_idx[k]] -= w * pair_dirs[k]
    
    # Wall distance gradients
    wall_weights = weights[N * (N - 1) // 2:]
    for i in range(N):
        w = wall_weights[i]
        if w > 1e-15:
            if wx[i] <= wy[i]:
                # dist = 0.5 - |x_i|
                if c[i, 0] >= 0:
                    grad_c[i, 0] += w
                else:
                    grad_c[i, 0] -= w
            else:
                # dist = 0.5 - |y_i|
                if c[i, 1] >= 0:
                    grad_c[i, 1] += w
                else:
                    grad_c[i, 1] -= w
    
    # f = -soft_min/2, so grad_f = 0.5 * grad(soft_min) = 0.5 * (-grad_c) ... 
    # Wait: soft_min = min - log(sum)/beta
    # d(soft_min)/d(c_i) = -sum_k w_k * d(dist_k)/d(c_i)
    # We computed grad_c[i] = sum_k w_k * d(dist_k)/d(c_i)
    # So d(soft_min)/d(c_i) = -grad_c[i]
    # f = -soft_min/2, so df/d(c_i) = -1/2 * d(soft_min)/d(c_i) = -1/2 * (-grad_c[i]) = 0.5 * grad_c[i]
    
    g = (0.5 * grad_c).flatten()
    
    return f, g

def _scipy_polish(centers, t0, deadline, maxiter=200):
    """
    Polish using scipy L-BFGS-B with smooth soft-min objective.
    """
    if time.time() - t0 > deadline - 10.0:
        return centers
    
    x0 = centers.flatten().copy()
    
    # Clip to valid range
    x0 = np.clip(x0, -0.5, 0.5)
    
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        result = minimize(
            _soft_min_obj_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        x_opt = result.x
        # Ensure valid
        if np.all(np.isfinite(x_opt)):
            c_opt = x_opt.reshape(N, 2)
            c_opt = np.clip(c_opt, -0.5, 0.5)
            return c_opt
    except Exception:
        pass
    
    return centers

def _scipy_polish_multi(centers, t0, deadline):
    """
    Try multiple beta values for the soft-min to escape local minima.
    """
    best_c = centers.copy()
    best_r = _score(centers)
    
    betas = [50.0, 80.0, 120.0]
    
    for beta in betas:
        if time.time() - t0 > deadline - 8.0:
            break
        
        x0 = centers.flatten().copy()
        x0 = np.clip(x0, -0.5, 0.5)
        
        def obj_and_grad(x, beta=beta):
            c = x.reshape(N, 2)
            
            # Pair distances (upper triangle)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=-1)
            mask = ~np.eye(N, dtype=bool)
            dists = np.sqrt(dists_sq[mask])
            
            # Wall distances
            wx = 0.5 - np.abs(c[:, 0])
            wy = 0.5 - np.abs(c[:, 1])
            wall_dists = np.minimum(wx, wy)
            
            all_dists = np.concatenate([dists, wall_dists])
            
            min_val = np.min(all_dists)
            exp_vals = np.exp(-beta * (all_dists - min_val))
            sum_exp = np.sum(exp_vals)
            soft_min = min_val - np.log(sum_exp) / beta
            weights = exp_vals / sum_exp
            
            f = -soft_min / 2.0
            
            grad_c = np.zeros((N, 2))
            
            pair_weights = weights[:N * (N - 1) // 2]
            i_idx, j_idx = np.where(mask)
            pair_diff = c[i_idx] - c[j_idx]
            pair_dists = dists
            safe_dists = np.maximum(pair_dists, 1e-12)
            pair_dirs = pair_diff / safe_dists[:, np.newaxis]
            
            for k in range(len(pair_weights)):
                w = pair_weights[k]
                if w > 1e-15:
                    grad_c[i_idx[k]] += w * pair_dirs[k]
                    grad_c[j_idx[k]] -= w * pair_dirs[k]
            
            wall_weights = weights[N * (N - 1) // 2:]
            for i in range(N):
                w = wall_weights[i]
                if w > 1e-15:
                    if wx[i] <= wy[i]:
                        if c[i, 0] >= 0:
                            grad_c[i, 0] += w
                        else:
                            grad_c[i, 0] -= w
                    else:
                        if c[i, 1] >= 0:
                            grad_c[i, 1] += w
                        else:
                            grad_c[i, 1] -= w
            
            g = (0.5 * grad_c).flatten()
            return f, g
        
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            x_opt = result.x
            if np.all(np.isfinite(x_opt)):
                c_opt = x_opt.reshape(N, 2)
                c_opt = np.clip(c_opt, -0.5, 0.5)
                r_opt = _score(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()
        except Exception:
            pass
    
    return best_c, best_r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 30) / n_inits  # reserve 30s for polish
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1200, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        # Check time
        if time.time() - t0 > DEADLINE - 35:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Langevin refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=6000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: Final fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Phase 4: L-BFGS polish with soft-min objective (structural improvement)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c, r = _scipy_polish_multi(best, t0, DEADLINE)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 5: One more Langevin pass if time allows
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=4000, T0=0.0008, T1=0.00002, lr=0.002)
        if r > best_r:
            best_r = r
            best = c
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
