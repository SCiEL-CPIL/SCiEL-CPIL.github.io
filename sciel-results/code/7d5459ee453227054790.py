"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin before the 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _compute_radius(pos):
    """Vectorized packing radius: min(min_pair/2, min_boundary)."""
    n = len(pos)
    if n == 0:
        return 0.0
    # Boundary constraint: distance from each centre to the nearest side.
    dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
    min_boundary = float(np.min(dist_boundary))
    if min_boundary <= 0.0:
        return 0.0
    # Pairwise distances via squared distances.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = math.sqrt(float(np.min(dist_sq)))
    return min(min_pair * 0.5, min_boundary)


def _generate_hex_candidates(spacing):
    """Dense hexagonal lattice of candidate centres inside [-0.5,0.5]^2."""
    dy = spacing * math.sqrt(3.0) / 2.0
    max_coord = 0.5
    points = []
    row = 0
    y = -max_coord
    while y <= max_coord:
        x_offset = 0.0 if (row % 2 == 0) else spacing / 2.0
        x = -max_coord + x_offset
        while x <= max_coord:
            if -max_coord <= x <= max_coord and -max_coord <= y <= max_coord:
                points.append((x, y))
            x += spacing
        y += dy
        row += 1
    return np.array(points, dtype=float) if points else np.zeros((0, 2))


def _soft_min(x, k=10.0):
    """
    Soft minimum of an array using log-sum-exp trick.
    Returns a smooth approximation to min(x).
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    # For soft min, we use -soft_max(-x)
    # soft_max(x) = (1/k) * log(sum(exp(k*x)))
    # soft_min(x) = -soft_max(-x) = -(1/k) * log(sum(exp(-k*x)))
    x_max = np.max(x)
    # To avoid overflow, shift by x_max
    # soft_min = x_max - (1/k) * log(sum(exp(k*(x - x_max))))
    # Wait, that's for soft_max.
    # Let's derive:
    # min(x) approx -1/k * log(sum(exp(-k*x)))
    # Let m = min(x). Then -k*x >= -k*m.
    # exp(-k*x) <= exp(-k*m).
    # sum(exp(-k*x)) <= n * exp(-k*m).
    # log(sum) <= log(n) - k*m.
    # -1/k * log(sum) >= -1/k * (log(n) - k*m) = m - log(n)/k.
    # So it approximates m from below? No, it's an upper bound on min?
    # Actually, soft_min is a smooth under-estimator or over-estimator?
    # As k -> inf, it converges to min.
    # For finite k, it is larger than min (since log(sum) > log(max term) = log(exp(-k*min)) = -k*min).
    # So -1/k * log(sum) < min.
    # It is a lower bound.
    
    # Numerically stable version:
    # val = -1/k * log(sum(exp(-k*x)))
    # Let y = -k*x.
    # val = 1/k * log(sum(exp(y)))
    # y_max = max(y) = -k*min(x)
    # sum(exp(y)) = exp(y_max) * sum(exp(y - y_max))
    # log(sum(exp(y))) = y_max + log(sum(exp(y - y_max)))
    # val = 1/k * (y_max + log(sum(exp(y - y_max))))
    # val = 1/k * (-k*min(x) + log(sum(exp(-k*(x - min(x))))))
    # val = -min(x) + 1/k * log(sum(exp(-k*(x - min(x)))))
    
    # This is stable.
    min_x = np.min(x)
    shifted = x - min_x
    # exp(-k * shifted)
    # shifted >= 0, so -k*shifted <= 0, exp is <= 1. No overflow.
    s = np.sum(np.exp(-k * shifted))
    if s == 0:
        return min_x
    return min_x + (1.0 / k) * np.log(s)


def _objective_and_grad(x_flat, k=10.0):
    """
    Objective: minimize -soft_min(constraints)
    Constraints:
    1. Boundary distances: 0.5 - |x_i|, 0.5 - |y_i| for each i. (2N values)
    2. Pairwise distances / 2: |x_i - x_j| / 2 for i < j. (N*(N-1)/2 values)
    
    We want to maximize the minimum of these.
    So we minimize -soft_min(constraints).
    
    x_flat: (2N,) array.
    """
    n = len(x_flat) // 2
    pos = x_flat.reshape(n, 2)
    
    # 1. Boundary constraints
    # dist_boundary[i] = min(0.5 - |x_i|, 0.5 - |y_i|)
    # But soft_min works on a flat array.
    # Let's collect all boundary distances.
    # For each point, dx = 0.5 - |x|, dy = 0.5 - |y|.
    # We can just take all 2N values.
    dx = 0.5 - np.abs(pos[:, 0])
    dy = 0.5 - np.abs(pos[:, 1])
    boundary_dists = np.concatenate([dx, dy])
    
    # 2. Pairwise distances
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # dist_sq = np.sum(diff * diff, axis=2)
    # dists = np.sqrt(dist_sq)
    # We only need i < j.
    # To avoid computing full matrix repeatedly in gradient, we can vectorize.
    
    # For soft_min, we need the values and the gradient.
    # Let z be the array of all constraint values (boundary + pair distances/2).
    # soft_min(z) = m + (1/k) * log(sum(exp(-k*(z - m)))) where m = min(z).
    # Let w = z - m.
    # g = sum(exp(-k*w))
    # f = m + (1/k) * log(g)
    # df/dz_i = (1/k) * (1/g) * (-k * exp(-k*w_i) * (1 - delta_mi)) 
    #          = -exp(-k*w_i)/g * (1 - delta_mi)
    # If z_i is the unique minimum, w_i = 0, exp(0)=1. g >= 1.
    # df/dz_i = -1/g.
    # For other j, df/dz_j = -exp(-k*w_j)/g.
    
    # However, m = min(z) is non-smooth.
    # The derivative of m is 0 almost everywhere, but undefined at ties.
    # In practice, soft_min is differentiable everywhere.
    # The formula above holds.
    
    # Let's compute z.
    # Boundary: 2N values.
    # Pairs: N*(N-1)/2 values.
    
    # Efficient computation of pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dist_sq[iu]
    pair_dists = np.sqrt(pair_dists_sq)
    pair_constraints = pair_dists * 0.5
    
    z = np.concatenate([boundary_dists, pair_constraints])
    
    # Compute soft_min
    m = np.min(z)
    w = z - m
    # Avoid underflow for large k*w
    # exp(-k*w)
    # If k*w is very large, exp is 0.
    # Use np.exp with clipping?
    # max_w = np.max(w)
    # shifted_w = w - max_w  # This makes exp <= 1
    # Actually, w >= 0.
    # If we shift by max_w, we get negative exponents.
    # Let's just use np.exp(-k * w).
    # To be safe against overflow in log, we can do:
    # log_sum = np.log(np.sum(np.exp(-k * w)))
    # This is safe if k*w doesn't overflow. k=10, w can be ~1. k*w=10. exp(-10) is small.
    # If w is large, exp is 0. Sum is at least 1 (for the min element).
    
    exp_terms = np.exp(-k * w)
    g = np.sum(exp_terms)
    if g == 0:
        g = 1.0  # Should not happen
    
    soft_min_val = m + (1.0 / k) * np.log(g)
    
    # Objective is -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    # d(soft_min)/dz_i = -exp_terms[i] / g
    # d(obj)/dz_i = exp_terms[i] / g
    
    grad_z = exp_terms / g
    
    # Now map grad_z back to x_flat.
    grad_x = np.zeros_like(x_flat)
    
    # 1. Boundary gradients
    # z[0:N] are dx = 0.5 - |x_i|
    # z[N:2N] are dy = 0.5 - |y_i|
    
    # d(dx)/dx_i = -sign(x_i)
    # d(dy)/dy_i = -sign(y_i)
    
    signs_x = np.sign(pos[:, 0])
    signs_y = np.sign(pos[:, 1])
    
    # grad_x[:, 0] += grad_z[0:N] * (-signs_x)
    # grad_x[:, 1] += grad_z[N:2N] * (-signs_y)
    
    grad_x[:, 0] -= grad_z[0:n] * signs_x
    grad_x[:, 1] -= grad_z[n:2*n] * signs_y
    
    # 2. Pair gradients
    # z[2N:] are pair_dists/2.
    # d(pair_dist/2)/dx_i = (x_i - x_j) / (2 * pair_dist) * 0.5 = (x_i - x_j) / (4 * pair_dist)
    # d(pair_dist/2)/dx_j = (x_j - x_i) / (4 * pair_dist)
    
    # We have pair_dists and indices iu.
    # iu[0] are i, iu[1] are j.
    # diff_ij = pos[i] - pos[j]
    # grad_pos[i] += grad_z_pair * (diff_ij / (4 * pair_dist))
    # grad_pos[j] -= grad_z_pair * (diff_ij / (4 * pair_dist))
    
    # Vectorized update
    # We need to accumulate into grad_x.
    # This is tricky with numpy for sparse updates.
    # Since N=200, number of pairs is ~20k.
    # We can use np.add.at
    
    # grad_z_pairs = grad_z[2*n:]
    # diff_ij = pos[iu[0]] - pos[iu[1]]
    # coeff = grad_z_pairs[:, np.newaxis] / (4.0 * pair_dists[:, np.newaxis])
    # grads = diff_ij * coeff
    
    # np.add.at(grad_x, iu[0], grads)
    # np.add.at(grad_x, iu[1], -grads)
    
    grad_z_pairs = grad_z[2*n:]
    # Handle zero distances to avoid division by zero
    safe_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = grad_z_pairs[:, np.newaxis] / (4.0 * safe_dists[:, np.newaxis])
    # If dist is 0, coeff is 0?
    # If dist is 0, pair_dist is 0.
    # The gradient is undefined, but numerically we can set it to 0.
    # The term exp(-k*w) might be non-zero if it's the minimum.
    # If two points coincide, it's a singularity.
    # Let's assume they don't coincide exactly.
    
    diff_ij = pos[iu[0]] - pos[iu[1]]
    grads = diff_ij * coeff
    
    np.add.at(grad_x, iu[0], grads)
    np.add.at(grad_x, iu[1], -grads)
    
    return obj, grad_x


def _generate_hex_initial(spacing):
    """Generate a hexagonal lattice and select N points."""
    candidates = _generate_hex_candidates(spacing)
    if len(candidates) < N:
        # Fallback to random
        return np.random.uniform(-0.49, 0.49, (N, 2))
    
    # We need to select N points from candidates.
    # Simplest: just take the first N? Or the ones closest to center?
    # Or use greedy?
    # For initialization, taking a subset that covers the square well is good.
    # Let's just take a random subset or the first N.
    # Actually, the lattice is regular. Taking the first N might be biased.
    # Let's take the N points closest to the center?
    # Or just shuffle and take N?
    # Shuffling is fine.
    idx = np.random.choice(len(candidates), N, replace=False)
    return candidates[idx]


def _run_optimization(pos_init, time_limit=100.0):
    """
    Run BFGS optimization on the soft-min objective.
    """
    start_time = time.time()
    pos = np.array(pos_init, dtype=float, copy=True)
    
    # Flatten
    x0 = pos.flatten()
    
    # Bounds: [-0.5, 0.5] for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_r = _compute_radius(pos)
    best_pos = pos.copy()
    
    # We can run multiple restarts or just one long one?
    # BFGS is local.
    # Let's try a few different k values?
    # Or just one with k=10.
    
    k = 10.0
    
    # Define objective for scipy
    def obj_func(x):
        val, _ = _objective_and_grad(x, k)
        return val
    
    def grad_func(x):
        _, g = _objective_and_grad(x, k)
        return g
    
    # Run minimize
    # maxiter needs to be large enough, but we check time.
    # scipy minimize doesn't have a time limit callback easily.
    # We can wrap the function to check time?
    # Or just run it and hope it finishes?
    # Better: run in a loop with small maxiter and check time?
    # Or use a callback to break?
    # scipy.optimize.minimize accepts a callback.
    
    state = {'done': False}
    
    def callback(xk):
        if _time_left() < 10.0:
            state['done'] = True
        # Also update best if we want
        # But callback is called after each iteration.
        # We can compute r here?
        # It might be slow.
        # Let's just let it finish or hit time limit.
        # Actually, if we don't break, it might run too long.
        # Let's check time in the objective function?
        # No, that's bad practice.
        
        # Alternative: run minimize with a reasonable maxiter and check time after.
        # But if one run takes 150s, we are dead.
        # Let's use a wrapper that raises an exception if time is up?
        # Or just rely on maxiter.
        # N=200, 400 variables. BFGS should be fast.
        # Let's set maxiter=1000.
        return state['done']
    
    try:
        res = minimize(
            obj_func,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50},
            callback=callback
        )
        pos_opt = res.x.reshape(N, 2)
        r_opt = _compute_radius(pos_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_pos = pos_opt
            
    except Exception:
        pass
    
    return best_pos, best_r


def _force_directed_phase(pos, time_limit=10.0):
    """
    Simple force-directed repulsion to spread points.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    start_time = time.time()
    
    while _time_left() > 5.0 and (time.time() - start_time) < time_limit:
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid div by 0
        dist = np.sqrt(dist_sq)
        
        # Force ~ 1/dist^2
        # F_ij = k * (pos_i - pos_j) / dist^3
        # Let's use a simple potential.
        
        # Vectorized force calculation is heavy for N=200 (40k pairs).
        # Let's do it in chunks or just accept the cost.
        
        inv_dist_sq = 1.0 / dist_sq
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        inv_dist_sq[~mask] = inv_dist_sq[~mask]
        inv_dist_sq[mask] = 0.0
        
        # F_i = sum_j (pos_i - pos_j) / dist_ij^3
        # = sum_j (pos_i - pos_j) * inv_dist_sq / dist_ij
        # = sum_j (pos_i - pos_j) * inv_dist_sq * inv_dist
        
        inv_dist = np.sqrt(inv_dist_sq)
        
        # Weight matrix W_ij = 1/dist^3
        # W = inv_dist_sq * inv_dist
        W = inv_dist_sq * inv_dist
        W[~mask] = W[~mask]
        W[mask] = 0.0
        
        # F_i = sum_j W_ij * (pos_i - pos_j)
        # F_i = pos_i * sum_j W_ij - sum_j W_ij * pos_j
        sum_W = np.sum(W, axis=1)
        forces += pos * sum_W[:, np.newaxis]
        forces -= W @ pos
        
        # Boundary forces
        # Push away from walls if close
        # x < -0.45: force_x += ( -0.45 - x ) * k
        # x > 0.45: force_x -= ( x - 0.45 ) * k
        # Let's use a soft wall.
        
        # Simple: if x < -0.49, push right.
        # Let's just clip and let optimization handle it.
        # Or add a small force towards center.
        
        # Update
        step_size = 0.001
        pos += forces * step_size
        pos = np.clip(pos, -0.499, 0.499)
        
    return pos


def _polish_pass(pos, max_iter=100):
    """
    High-precision polish: iteratively find the closest pair and boundary,
    then move points to increase the minimum distance.
    """
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    
    for it in range(max_iter):
        if _time_left() < 8.0:
            break
        
        r = _compute_radius(pos)
        if r <= 0:
            break
        
        # Find the minimum pair and boundary
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        min_pair_dist = math.sqrt(dist_sq[min_idx])
        
        dist_boundary = np.min(0.5 - np.abs(pos), axis=1)
        min_bnd_idx = int(np.argmin(dist_boundary))
        min_bnd_dist = dist_boundary[min_bnd_idx]
        
        # Determine which constraint is active
        if min_pair_dist / 2.0 <= min_bnd_dist:
            # Pair constraint is active
            i, j = min_idx
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.random.randn(2) * 1e-8
                dist = np.linalg.norm(d)
            d_norm = d / dist
            # Move i and j apart by a small amount
            step = 0.01 * (dist - 2 * r)
            if step > 0:
                pos[i] += d_norm * step / 2
                pos[j] -= d_norm * step / 2
        else:
            # Boundary constraint is active
            p = pos[min_bnd_idx]
            # Move toward center
            step = 0.01 * (min_bnd_dist - r)
            if step > 0:
                norm_p = np.linalg.norm(p)
                if norm_p > 1e-10:
                    pos[min_bnd_idx] = p + (-p / norm_p) * step
                else:
                    pos[min_bnd_idx] += np.random.randn(2) * step

        pos = np.clip(pos, -0.49999, 0.49999)
        
        # Check if we made progress
        new_r = _compute_radius(pos)
        if new_r <= r + 1e-10:
            # No progress, try a small random perturbation
            pos += np.random.randn(n, 2) * 0.0001
            pos = np.clip(pos, -0.49999, 0.49999)
    
    return pos


def construct():
    best_pos = None
    best_r = 0.0

    # 1) Generate initial positions from hexagonal lattices
    spacings = [0.035, 0.03, 0.025]
    for sp in spacings:
        if _time_left() < 30.0:
            break
        try:
            pos_init = _generate_hex_initial(sp)
            # Run optimization
            opt_pos, opt_r = _run_optimization(pos_init, time_limit=_time_left() - 10.0)
            if opt_r > best_r:
                best_r = opt_r
                best_pos = opt_pos.copy()
        except Exception:
            continue

    # Fallback: coarse uniform grid
    if best_pos is None or best_r < 0.02:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best_pos = grid.copy()

    if best_pos is None:
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_r = _compute_radius(best_pos)

    # 2) Force-directed phase to improve spread
    if _time_left() > 15.0:
        fd_pos = _force_directed_phase(best_pos, time_limit=min(10.0, _time_left() - 10.0))
        r_fd = _compute_radius(fd_pos)
        if r_fd > best_r:
            best_r = r_fd
            best_pos = fd_pos.copy()
        
        # Run optimization again on the force-directed result
        if _time_left() > 15.0:
            opt_pos2, opt_r2 = _run_optimization(best_pos, time_limit=_time_left() - 10.0)
            if opt_r2 > best_r:
                best_r = opt_r2
                best_pos = opt_pos2.copy()

    # 3) Polish pass
    if _time_left() > 10.0:
        polished = _polish_pass(best_pos, max_iter=100)
        r3 = _compute_radius(polished)
        if r3 > best_r:
            best_r = r3
            best_pos = polished

    # Final safety
    best_pos = np.asarray(best_pos, dtype=float)
    if best_pos.shape != (N, 2):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
        best_pos = grid
    best_pos = np.clip(best_pos, -0.49999, 0.49999)
    if not np.all(np.isfinite(best_pos)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return best_pos
# EVOLVE-BLOCK-END
