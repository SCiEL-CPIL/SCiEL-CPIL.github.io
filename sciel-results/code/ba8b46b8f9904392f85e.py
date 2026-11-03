"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
import math
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

def _fast_radius(c, R):
    """Fast check: is the configuration valid at radius R? Returns min margin."""
    # Wall check
    wall = 0.5 - np.abs(c).max(axis=1)
    r_wall = wall.min()
    if r_wall < R:
        return r_wall - R
    # Pair check - use squared distances for speed
    diff = c[:, np.newaxis, :] - c[np.newaxis, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = dist_sq.min()
    min_dist = np.sqrt(min_dist_sq)
    return min_dist / 2.0 - R

def _fast_score(c):
    """Compute radius without the expensive pairwise sqrt - only when needed."""
    pos = np.asarray(c, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    Uses fast radius check to reduce expensive _score calls.
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    score_interval = 8  # only do full score every N steps
    step_count = 0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        step_count += 1
        # Fast check first
        fast_r = _fast_radius(c, R)
        if fast_r < -0.002:
            # Configuration is badly invalid, skip scoring
            R += step
            continue
        # Periodic full score - use faster _fast_score
        if step_count % score_interval == 0 or R >= R1 - step:
            r = _fast_score(c)
            if r > best[1]:
                best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    # Final score
    r = _score(c)
    if r > best[1]:
        best = (c.copy(), r)
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

def _init_hex_tight():
    """Tighter hexagonal lattice."""
    positions = []
    spacing = 0.135
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(8):
            x = (col - 3.5) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_grid_10x10_pick(rng):
    """10x10 grid, randomly pick 50."""
    ticks = (np.arange(10) + 0.5) / 10 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    return g[idx].copy()

def _init_clustered(rng):
    """Two clusters of 25 circles each."""
    c1 = rng.uniform(-0.4, -0.05, (25, 2))
    c2 = rng.uniform(0.05, 0.4, (25, 2))
    return np.vstack([c1, c2])

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    Uses fast scoring to reduce overhead.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    score_every = 20  # only score every N steps
    
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
        
        # Only score periodically to save time
        if step % score_every == 0 or step == steps - 1:
            r_new = _fast_score(c)
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
    
    # Final accurate score
    r_final = _score(best_c)
    if r_final > best_r:
        best_r = r_final
    
    return best_c, best_r

def _soft_min_radius(c, beta=50.0):
    """Compute soft-min of all distances (pairs and walls)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = np.min(np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ]), axis=0)
    
    # All distances
    all_dists = np.concatenate([dists[~np.eye(n, dtype=bool)], wall_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _soft_min_radius_and_grad(c, beta=50.0):
    """Compute soft-min radius and its gradient analytically (vectorized)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances (per circle, min of x and y wall distances)
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    
    # All distances: upper triangle pairs + wall distances
    # Use upper triangle indices
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min value
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights
    weights = exp_vals / sum_exp
    
    grad = np.zeros((n, 2))
    
    # Pair distance gradients (vectorized)
    pair_weights = weights[:len(pair_dists)]
    # diff for upper triangle pairs
    pair_diff = diff[iu]  # shape (n_pairs, 2)
    pair_d = dists[iu]  # shape (n_pairs,)
    
    # direction = (c_i - c_j) / d
    safe_d = np.maximum(pair_d, 1e-12)
    directions = pair_diff / safe_d[:, np.newaxis]  # shape (n_pairs, 2)
    
    # Weighted sum: for each pair, w * direction contributes to grad[i] -= w*dir, grad[j] += w*dir
    weighted_dirs = pair_weights[:, np.newaxis] * directions  # shape (n_pairs, 2)
    
    # Scatter: for each pair (i,j), add -w*dir to grad[i] and +w*dir to grad[j]
    # Using np.add.at for accumulated additions
    np.add.at(grad, iu[0], -weighted_dirs)
    np.add.at(grad, iu[1], weighted_dirs)
    
    # Wall distance gradients (vectorized)
    wall_weights = weights[len(pair_dists):]
    # For each circle, determine which wall is limiting
    is_x_limit = wall_x <= wall_y
    # x contribution: sign depends on c[i,0]
    x_sign = np.sign(c[:, 0])
    x_sign[x_sign == 0] = 1.0
    # y contribution
    y_sign = np.sign(c[:, 1])
    y_sign[y_sign == 0] = 1.0
    
    # grad[i,0] += w * x_sign if x is limiting, else 0
    # grad[i,1] += w * y_sign if y is limiting, else 0
    grad[:, 0] += np.where(is_x_limit, wall_weights * x_sign, 0.0)
    grad[:, 1] += np.where(~is_x_limit, wall_weights * y_sign, 0.0)
    
    return soft_min, grad

def _scipy_polish(centers, deadline=None):
    """Polish using scipy with smooth soft-min objective and analytical gradient."""
    if deadline is not None and time.time() > deadline - 8.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    best_result = None
    best_r = _score(centers)

    for beta in [30.0, 80.0, 200.0, 500.0]:
        if deadline is not None and time.time() > deadline - 4.0:
            break

        def objective(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            return -_soft_min_radius(c, beta)

        def objective_grad(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, grad = _soft_min_radius_and_grad(c, beta)
            return -val, -grad

        bounds = [(-0.5, 0.5)] * (2 * n)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=objective_grad,
                              options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 2000})
            polished = result.x.reshape(n, 2)
            r = _score(polished)
            if r > best_r:
                best_r = r
                best_result = polished
                x0 = result.x.copy()
        except Exception:
            pass

    if best_result is not None and best_r >= _score(centers) - 1e-10:
        return best_result

    return centers

def _contact_graph_refine(c, r, t0, deadline, max_iters=100):
    """
    Active-set refinement: identify circles in contact (distance <= 2r + tol)
    and solve for exact contact geometry using Newton's method on the
    constraint equations. Vectorized for speed.
    """
    if deadline is not None and time.time() > deadline - 3.0:
        return c
    
    c = c.copy()
    n = len(c)
    tol = 1e-7
    
    for outer_iter in range(max_iters):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        # Identify contact pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        
        # Contact pairs: distance within tolerance of 2r
        contact_mask = dist < (2 * r + tol)
        np.fill_diagonal(contact_mask, False)
        
        # Wall contacts
        wall_contact = np.zeros(n, dtype=bool)
        wall_contact |= (0.5 - np.abs(c[:, 0])) < (r + tol)
        wall_contact |= (0.5 - np.abs(c[:, 1])) < (r + tol)
        
        # Vectorized pair corrections
        iu = np.triu_indices(n, k=1)
        contact_pairs = np.where(contact_mask[iu])
        if len(contact_pairs[0]) > 0:
            i_idx = iu[0][contact_pairs]
            j_idx = iu[1][contact_pairs]
            pair_diff = c[i_idx] - c[j_idx]
            pair_dist = np.sqrt((pair_diff ** 2).sum(axis=1))
            mask_valid = pair_dist > 1e-12
            if np.any(mask_valid):
                scale = (2 * r) / pair_dist[mask_valid]
                corrections = (scale - 1) * pair_diff[mask_valid] * 0.5
                np.add.at(c, i_idx[mask_valid], corrections)
                np.add.at(c, j_idx[mask_valid], -corrections)
        
        # Wall corrections (vectorized)
        if np.any(wall_contact):
            wc = c[wall_contact].copy()
            wc[:, 0] = np.clip(wc[:, 0], -0.5 + r, 0.5 - r)
            wc[:, 1] = np.clip(wc[:, 1], -0.5 + r, 0.5 - r)
            c[wall_contact] = wc
        
        c = np.clip(c, -0.5, 0.5)
    
    return c

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
        _init_hex_tight(),
        _init_grid_10x10_pick(np.random.default_rng(123)),
        _init_clustered(np.random.default_rng(456)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    # Allocate more time to growth since it's the main search phase
    n_inits = len(inits)
    growth_deadline = DEADLINE - 40  # reserve 40s for refinement
    
    for i, start in enumerate(inits):
        remaining = growth_deadline - (time.time() - t0)
        if remaining < 8:
            break
        
        # Grow from 0.05 to 0.072 with slightly larger steps for speed
        c, r = _grow(start, 0.05, 0.072, 0.0004, 800, 0.025, 0.003, t0, growth_deadline)
        
        if r > best_r:
            best_r = r
            best = c.copy()
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Langevin refinement on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=12000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: Analytic gradient polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c_polished = _scipy_polish(best, DEADLINE)
        r_polished = _score(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best = c_polished
    
    # Phase 4: Contact graph refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        c_contact = _contact_graph_refine(best, best_r, t0, DEADLINE)
        r_contact = _score(c_contact)
        if r_contact > best_r:
            best_r = r_contact
            best = c_contact
    
    # Phase 5: Fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        r_cur = _score(best)
        c = _relax(best, r_cur, 3000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Phase 6: Second Langevin pass with lower temperature
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=10000, T0=0.0008, T1=0.00002, lr=0.002)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 7: Final analytic polish with tighter settings
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 6:
        c_polished = _scipy_polish(best, DEADLINE)
        r_polished = _score(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best = c_polished
    
    # Phase 8: Final micro-relaxation
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 4:
        r_cur = _score(best)
        c = _relax(best, r_cur, 1500, 0.005, 0.0002, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
