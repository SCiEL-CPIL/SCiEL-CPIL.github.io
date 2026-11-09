"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s limit
_start_time = time.time()


def _time_left():
    return DEADLINE - (time.time() - _start_time)


def _compute_radius(centers):
    """Compute the score: largest common radius for given centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _softmin_value_and_grad(centers, beta=200.0):
    """
    Compute the smooth soft-min surrogate and its gradient.
    
    r_soft = -beta * log( sum_{pairs} exp(-d_ij/(2*beta)) + sum_i exp(-w_i/beta) )
    
    where d_ij are pairwise distances and w_i are wall distances.
    This is a smooth approximation to min(min_pair/2, min_wall).
    """
    centers = np.asarray(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Only consider upper triangle for pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Soft-min: -beta * log(sum exp(-x_i/beta))
    # For numerical stability, shift by min
    all_vals = np.concatenate([pair_dists / 2.0, wall_dists])
    min_val = np.min(all_vals)
    
    # exp terms
    exp_terms = np.exp(-(all_vals - min_val) / beta)
    log_sum = np.log(np.sum(exp_terms))
    r_soft = min_val - beta * log_sum
    
    # Gradient: d r_soft / d centers
    # d r_soft / d x_i = -exp(-x_i/beta) / sum_j exp(-x_j/beta) * d x_i / d centers
    # For pair (i,j): x = d_ij/2, d x / d c_i = (c_i - c_j) / (2 * d_ij)
    # For wall i: x = w_i, d x / d c_i depends on which wall
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    # weights for pairs
    pair_exp = np.exp(-(pair_dists / 2.0 - min_val) / beta)
    pair_weights = pair_exp / np.sum(exp_terms)
    
    # For each pair (i,j), weight * (c_i - c_j) / (2 * d_ij) added to c_i
    # and weight * (c_j - c_i) / (2 * d_ij) added to c_j
    # i.e., weight * (c_i - c_j) / (2 * d_ij) * (1 for i, -1 for j)
    
    # Vectorized: for each pair, add to grad[i] and grad[j]
    # grad[i] += sum_j weight_ij * (c_i - c_j) / (2 * d_ij)
    # grad[j] -= sum_i weight_ij * (c_i - c_j) / (2 * d_ij)
    
    # Build force vectors
    # For pair (i,j): direction = (c_i - c_j) / d_ij, magnitude = weight / 2
    # Force on i: +weight/2 * (c_i - c_j) / d_ij
    # Force on j: -weight/2 * (c_i - c_j) / d_ij
    
    # Use the upper triangle indices
    i_idx, j_idx = iu
    pair_dirs = (centers[i_idx] - centers[j_idx]) / (pair_dists[:, np.newaxis] + 1e-15)
    pair_forces = (pair_weights / 2.0)[:, np.newaxis] * pair_dirs
    
    # Accumulate using bincount-like approach
    # For each pair, add pair_force to i and subtract from j
    for k in range(len(i_idx)):
        grad[i_idx[k]] += pair_forces[k]
        grad[j_idx[k]] -= pair_forces[k]
    
    # Wall contributions
    wall_exp = np.exp(-(wall_dists - min_val) / beta)
    wall_weights = wall_exp / np.sum(exp_terms)
    
    # For each center, determine which wall is closest
    for i in range(N):
        ci = centers[i]
        # Distance to each wall: left, right, bottom, top
        d_left = ci[0] + 0.5
        d_right = 0.5 - ci[0]
        d_bottom = ci[1] + 0.5
        d_top = 0.5 - ci[1]
        
        # The wall distance is min of these
        # Gradient of wall_dist w.r.t. c_i:
        # If left wall is min: d/dx = 1, d/dy = 0
        # If right wall is min: d/dx = -1, d/dy = 0
        # If bottom wall is min: d/dx = 0, d/dy = 1
        # If top wall is min: d/dx = 0, d/dy = -1
        
        min_d = min(d_left, d_right, d_bottom, d_top)
        w = wall_weights[i]
        
        if d_left <= d_right and d_left <= d_bottom and d_left <= d_top:
            grad[i, 0] += w  # d wall_dist / d x = 1 for left wall
        elif d_right <= d_left and d_right <= d_bottom and d_right <= d_top:
            grad[i, 0] -= w  # d wall_dist / d x = -1 for right wall
        elif d_bottom <= d_left and d_bottom <= d_right and d_bottom <= d_top:
            grad[i, 1] += w  # d wall_dist / d y = 1 for bottom wall
        else:
            grad[i, 1] -= w  # d wall_dist / d y = -1 for top wall
    
    return r_soft, grad


def _gradient_ascent_smooth(centers, max_iter=3000, beta_start=50.0, beta_end=500.0):
    """
    Gradient ascent on the smooth soft-min objective with adaptive beta.
    """
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    step_size = 0.01
    min_step = 1e-10
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Adaptive beta: start small (closer to true min), increase for smoothness
        progress = it / max(max_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        r_soft, grad = _softmin_value_and_grad(centers, beta=beta)
        
        # Gradient ascent
        new_centers = centers + step_size * grad
        new_centers = np.clip(new_centers, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(new_centers)
        
        if new_r > current_r + 1e-12:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            # Reset step size on improvement
            step_size = min(step_size * 1.2, 0.05)
        else:
            step_size *= 0.5
            if step_size < min_step:
                break
    
    return best_centers, best_r


def _high_precision_polish(centers, max_iter=5000):
    """
    High-precision polish using scipy minimize with tight tolerances.
    Maximizes the soft-min surrogate with L-BFGS-B.
    """
    centers = np.asarray(centers).copy()
    
    def neg_softmin(c, beta=500.0):
        r, _ = _softmin_value_and_grad(c, beta=beta)
        return -r
    
    def neg_softmin_grad(c, beta=500.0):
        _, g = _softmin_value_and_grad(c, beta=beta)
        return -g
    
    # Use multiple beta values for progressive refinement
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for beta in [200.0, 500.0, 1000.0]:
        if _time_left() < 10.0:
            break
        
        try:
            result = minimize(
                neg_softmin,
                best_centers,
                jac=neg_softmin_grad,
                method='L-BFGS-B',
                args=(beta,),
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': max_iter // len([200, 500, 1000]), 
                         'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            new_r = _compute_radius(result.x)
            if new_r > best_r:
                best_r = new_r
                best_centers = result.x.copy()
        except Exception:
            pass
    
    return best_centers, best_r


def _hex_initial():
    """Hexagonal packing initialization."""
    centers = np.zeros((N, 2))
    rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
    spacing = 1.0 / (rows * 0.866)
    
    idx = 0
    for row in range(rows):
        if idx >= N:
            break
        y = -0.5 + spacing * (row + 0.5) * 0.866
        if abs(y) > 0.5:
            continue
        offset = (row % 2) * spacing * 0.5
        max_cols = int((1.0 - 2 * spacing * 0.25) / spacing) + 1
        for col in range(max_cols):
            if idx >= N:
                break
            x = -0.5 + spacing * (col + 0.5) + offset
            if abs(x) <= 0.5:
                centers[idx] = [x, y]
                idx += 1
        if idx >= N:
            break
    
    if idx < N:
        for i in range(idx, N):
            centers[i] = [np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)]
    
    return centers[:N]


def _grid_initial(m):
    """Grid initialization with m columns."""
    centers = []
    n_rows = int(np.ceil(N / m))
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(n_rows) + 0.5) / n_rows - 0.5
    for y in ys:
        for x in xs:
            centers.append([x, y])
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    while len(centers) < N:
        centers.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    return np.array(centers[:N])


def _random_initial():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def construct():
    """
    Construct the centers of 200 equal circles inside [-1/2, 1/2]^2
    to maximize the common radius.
    """
    global _start_time
    _start_time = time.time()
    
    best_centers = None
    best_r = -1.0
    
    # Try multiple initializations
    initializations = []
    
    for m in [10, 11, 12, 13, 14]:
        initializations.append(_grid_initial(m))
    
    initializations.append(_hex_initial())
    
    for _ in range(3):
        initializations.append(_random_initial())
    
    # Process each initialization
    for init_idx, init_centers in enumerate(initializations):
        if _time_left() < 15.0:
            break
        
        r = _compute_radius(init_centers)
        if r > best_r:
            best_r = r
            best_centers = init_centers.copy()
        
        # Smooth gradient ascent
        try:
            opt_centers, opt_r = _gradient_ascent_smooth(init_centers, max_iter=1500)
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # High-precision polish on the best result
    if best_centers is not None and _time_left() > 20.0:
        try:
            polished_centers, polished_r = _high_precision_polish(best_centers, max_iter=3000)
            if polished_r > best_r:
                best_r = polished_r
                best_centers = polished_centers.copy()
        except Exception:
            pass
    
    # Final smooth gradient ascent pass
    if best_centers is not None and _time_left() > 10.0:
        try:
            final_centers, final_r = _gradient_ascent_smooth(best_centers, max_iter=2000)
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
