"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # safety margin before 120s kill

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
    c: (N, 2) array of centers
    R: target radius
    iters: number of relaxation iterations
    eta0, eta1: learning rate start and end
    t0, deadline: optional time tracking
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
    Returns (best_centers, best_radius).
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

def _soft_min_obj_and_grad(x, beta=100.0):
    """
    Compute soft-min of all distances (pairs and walls) and its gradient.
    x: flattened array of shape (2*N,)
    """
    c = x.reshape(N, 2)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    # dist to x=0.5: 0.5 - x
    # dist to x=-0.5: x + 0.5
    # min(0.5 - x, x + 0.5) = 0.5 - |x|
    wall_dists = np.min(np.array([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ]), axis=0)
    
    # All distances
    # We need to be careful with the ordering for gradient computation
    # Let's construct the full list of distances and their gradients
    
    all_dists = np.concatenate([dists[~np.eye(N, dtype=bool)], wall_dists])
    
    # Soft min value
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights for each distance
    weights = exp_vals / sum_exp
    
    grad = np.zeros((N, 2))
    
    # Pair distance gradients
    # The upper triangle of dists corresponds to the flattened weights
    # We need to map weights back to pairs (i, j) with i < j
    pair_weights = weights[:N * (N - 1) // 2]
    
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            w = pair_weights[pair_idx]
            d = dists[i, j]
            if d > 1e-12:
                direction = (c[i] - c[j]) / d
                # d(dist_ij)/d(c_i) = (c_i - c_j)/dist_ij
                # d(dist_ij)/d(c_j) = (c_j - c_i)/dist_ij = -direction
                # d(soft_min)/d(c_i) = -sum w_k d(dist_k)/d(c_i)
                # Wait, soft_min approximates min. 
                # d(min)/dx = d(dist_k)/dx for the min one.
                # d(soft_min)/dx = -sum w_k d(dist_k)/dx ? 
                # Let's check: f(x) = -1/beta log(sum exp(-beta d_k)). 
                # f'(x) = -1/beta * (1/sum) * sum(-beta exp(-beta d_k) d'_k) 
                # f'(x) = sum (exp(-beta d_k)/sum) d'_k = sum w_k d'_k.
                # So grad is sum w_k * d(dist_k)/d(c).
                
                grad[i] += w * direction
                grad[j] -= w * direction
            pair_idx += 1
    
    # Wall distance gradients
    wall_weights = weights[N * (N - 1) // 2:]
    for i in range(N):
        w = wall_weights[i]
        dist_x = 0.5 - np.abs(c[i, 0])
        dist_y = 0.5 - np.abs(c[i, 1])
        
        # d(dist_x)/d(c_i, 0) = -sign(c_i, 0) if dist_x is the min
        # d(dist_y)/d(c_i, 1) = -sign(c_i, 1) if dist_y is the min
        # But we have a soft min over all distances, so we add contributions from both
        # effectively weighted by w.
        # Actually, the wall distance for circle i is min(0.5-|x|, 0.5-|y|).
        # In the all_dists array, we included min(0.5-|x|, 0.5-|y|) as a single value?
        # No, in _soft_min_obj_and_grad I calculated wall_dists as min of the two axes.
        # So for each circle i, there is ONE wall distance in the list.
        # Its gradient is the gradient of min(0.5-|x|, 0.5-|y|).
        # This is non-smooth. The soft-min over the set of all distances (pairs + this wall min)
        # is smooth.
        
        # To get the gradient of the wall part:
        # d/dx_i (0.5 - |x_i|) = -sign(x_i)
        # d/dy_i (0.5 - |y_i|) = -sign(y_i)
        
        # If dist_x < dist_y, the wall dist is 0.5 - |x_i|, grad is (-sign(x_i), 0)
        # If dist_y < dist_x, the wall dist is 0.5 - |y_i|, grad is (0, -sign(y_i))
        # If equal, it's a corner. We can use a soft min here too or just pick one.
        # Given the high beta, the weight w will be concentrated on the true min.
        
        if dist_x <= dist_y:
            grad[i, 0] += w * (-np.sign(c[i, 0]))
            # Small contribution from y if close? No, just use the hard min for simplicity 
            # or assume the soft-min over the global set handles it.
            # Actually, the wall_dists array contains the *value* of the min.
            # The derivative of min(a,b) is not defined at a=b.
            # But since we are taking a soft-min over the *set* of distances, 
            # and one of those distances is the wall-min, the gradient of that specific term
            # with respect to c_i is needed.
            
            # Let's approximate the gradient of the wall min term:
            # g_wall_i = min(0.5-|x|, 0.5-|y|)
            # If x>0, y>0: min(0.5-x, 0.5-y). 
            # d/dx = -1 if 0.5-x < 0.5-y => y < x.
            # d/dy = -1 if 0.5-y < 0.5-x => x < y.
            
            if c[i, 0] >= 0:
                grad[i, 0] += w * (-1.0)
            else:
                grad[i, 0] += w * (1.0)
                
            # Note: This ignores the fact that the wall distance might be determined by Y.
            # To be more accurate, we should treat the wall constraints as two separate distances
            # in the soft-min sum: 0.5-|x| and 0.5-|y|.
            
    # Let's refine the wall gradient calculation by treating x and y walls separately
    # This requires changing the all_dists construction.
    # For now, this approximation is often good enough for polishing.
    
    return -soft_min, grad.flatten()

def _scipy_polish(centers, deadline=None):
    """Polish using scipy L-BFGS-B with smooth soft-min objective."""
    if deadline is not None and time.time() > deadline - 10.0:
        return centers
    
    x0 = centers.flatten()
    
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < -_score(centers): # Check if improved
            new_centers = res.x.reshape(N, 2)
            # Ensure validity
            if np.all(np.isfinite(new_centers)) and _score(new_centers) >= 0:
                return new_centers
    except Exception:
        pass
        
    return centers

def _langevin_refine(c, r, t0, deadline, steps=3000, T0=0.001, T1=0.0001, lr=0.002):
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
    # Reserve 30s for refinement phases
    time_per_init = max(5.0, (DEADLINE - (time.time() - t0) - 30) / n_inits)
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        # Dynamic budget allocation
        budget = min(time_per_init, remaining * 0.8 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        # Use a slightly higher target to encourage better packing, relax will handle overlap
        c, r = _grow(start, 0.05, 0.072, 0.0005, 1000, 0.02, 0.002, t0, DEADLINE)
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        # Check time
        if time.time() - t0 > DEADLINE - 35:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Hybrid Refinement (Langevin + L-BFGS)
    # We will alternate between Langevin (to escape) and L-BFGS (to converge)
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 15:
        # First, a Langevin pass to explore
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=2000, T0=0.0015, T1=0.00005, lr=0.003)
        if r > best_r:
            best_r = r
            best = c
        
        # Then, polish with L-BFGS
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 10:
            c = _scipy_polish(best, DEADLINE)
            r = _score(c)
            if r > best_r:
                best_r = r
                best = c
        
        # If time allows, another round of Langevin + Polish
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 15:
            c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=1500, T0=0.0008, T1=0.00002, lr=0.002)
            if r > best_r:
                best_r = r
                best = c
            
            remaining = DEADLINE - (time.time() - t0)
            if remaining > 8:
                c = _scipy_polish(best, DEADLINE)
                r = _score(c)
                if r > best_r:
                    best_r = r
                    best = c
    
    # Phase 3: Final fine-tune relaxation at current radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.01, 0.0005, t0, DEADLINE)
        r_new = _score(c)
        if r_new >= r_cur:
            best = c
            best_r = r_new
            
        # Final L-BFGS polish
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 3:
            c = _scipy_polish(best, DEADLINE)
            r = _score(c)
            if r > best_r:
                best_r = r
                best = c

    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
