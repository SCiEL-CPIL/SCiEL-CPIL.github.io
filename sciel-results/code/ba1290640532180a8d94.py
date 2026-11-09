"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START

def _true_radius(positions):
    """Compute the exact minimum radius (score)."""
    if len(positions) == 0:
        return 0.0
    
    # Boundary distances
    # min(0.5 - |x|, 0.5 - |y|)
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    
    if min_boundary <= 0:
        return 0.0

    # Pairwise distances
    # Use broadcasting for speed
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _soft_min_radius_and_grad(positions, beta=50.0):
    """
    Compute soft-min radius and its gradient.
    Returns (value, gradient).
    """
    n = len(positions)
    pos_flat = positions.reshape(-1)
    
    # Boundary terms
    # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # For soft min, we treat each boundary constraint separately or the min of the two?
    # The constraint is that the circle must be inside. The distance to boundary is 
    # min(0.5-|x|, 0.5-|y|). 
    # To be smooth, we can use soft-min on the four boundaries: 
    # 0.5-x, 0.5+x, 0.5-y, 0.5+y.
    # Actually, the distance to the boundary of the square for a point (x,y) is 
    # min(0.5-x, 0.5+x, 0.5-y, 0.5+y).
    
    b_0 = 0.5 - positions[:, 0]
    b_1 = 0.5 + positions[:, 0]
    b_2 = 0.5 - positions[:, 1]
    b_3 = 0.5 + positions[:, 1]
    
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_radii = dist / 2.0
    
    # Collect all constraints
    # We have 4*N boundary constraints and N*(N-1)/2 pairwise constraints.
    # For efficiency in gradient, we process them in chunks or vectorized.
    
    # Let's build the array of all radii
    all_radii = np.concatenate([
        b_0, b_1, b_2, b_3, 
        pair_radii[np.triu_indices(n, k=1)]
    ])
    
    # Soft min: -1/beta * log(sum(exp(-beta * r_i)))
    # To avoid overflow, use max shift
    max_r = np.max(all_radii)
    # If max_r is very large, -beta * r_i will be very negative, exp will underflow to 0.
    # We want to approximate min. 
    # soft_min = -1/beta * log(sum(exp(-beta * r_i)))
    # Let M = min(r_i). exp(-beta * M) is the largest term.
    # It's safer to compute:
    # log_sum_exp(-beta * r_i) = -beta * M + log(sum(exp(-beta * (r_i - M))))
    # soft_min = M - 1/beta * log(sum(exp(-beta * (r_i - M))))
    
    # However, we need the gradient.
    # d(soft_min)/d(p) = -1/beta * d(log_sum_exp)/d(p)
    # d(log_sum_exp(z_i))/d(p) = sum( (exp(z_i) / sum(exp(z_j))) * dz_i/d(p) )
    # Here z_i = -beta * r_i.
    # dz_i/d(p) = -beta * dr_i/d(p).
    # So d(soft_min)/d(p) = -1/beta * sum( w_i * (-beta * dr_i/d(p)) ) = sum( w_i * dr_i/d(p) )
    # where w_i = exp(z_i) / sum(exp(z_j)).
    # This is a weighted average of the gradients of the individual radii, weighted by their "closeness" to the minimum.
    
    # Compute weights
    neg_beta_r = -beta * all_radii
    # Shift by max for stability
    max_neg_beta_r = np.max(neg_beta_r)
    exp_vals = np.exp(neg_beta_r - max_neg_beta_r)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Value
    log_sum = np.log(sum_exp) + max_neg_beta_r
    soft_min_val = -log_sum / beta
    
    # Gradient
    grad = np.zeros(2 * n)
    
    # 1. Boundary gradients
    # Indices 0 to 4*N - 1 correspond to b_0, b_1, b_2, b_3 for each point
    # b_0 = 0.5 - x. d(b_0)/dx = -1
    # b_1 = 0.5 + x. d(b_1)/dx = +1
    # b_2 = 0.5 - y. d(b_2)/dy = -1
    # b_3 = 0.5 + y. d(b_3)/dy = +1
    
    # w_0 for point i is weights[i] (if we ordered them 0,1,2,3 per point? No, we concatenated arrays)
    # Let's stick to the order: all b_0, then all b_1, then all b_2, then all b_3.
    # b_0: indices 0..N-1. d/dx_i = -1.
    # b_1: indices N..2N-1. d/dx_i = +1.
    # b_2: indices 2N..3N-1. d/dy_i = -1.
    # b_3: indices 3N..4N-1. d/dy_i = +1.
    
    w_b0 = weights[0:N]
    w_b1 = weights[N:2*N]
    w_b2 = weights[2*N:3*N]
    w_b3 = weights[3*N:4*N]
    
    # grad_x gets -w_b0 + w_b1
    # grad_y gets -w_b2 + w_b3
    grad[0::2] += (-w_b0 + w_b1)
    grad[1::2] += (-w_b2 + w_b3)
    
    # 2. Pairwise gradients
    # Indices 4*N to end correspond to upper triangle of pair_radii
    # We need to map the flat index back to (i, j)
    # Number of pairs is N*(N-1)/2
    pair_weights = weights[4*N:]
    
    # Vectorized update for pairwise gradients is tricky without explicit loops or complex indexing.
    # For N=200, pairs = 19900. A loop is acceptable if it's not too slow, but vectorized is better.
    # Let's use the fact that grad[i] += sum_j w_ij * (p_i - p_j) / dist_ij
    # and grad[j] += sum_i w_ij * (p_j - p_i) / dist_ij
    
    # We can accumulate contributions using np.add.at
    # First, get indices i, j
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # dist is already computed. We need dist for the gradient direction.
    # dist[i, j]
    d = dist[i_idx, j_idx]
    
    # Avoid division by zero
    mask = d > 1e-12
    if not np.all(mask):
        d = np.where(mask, d, 1.0)
        pair_weights = np.where(mask, pair_weights, 0.0)
        
    # Vector = (p_i - p_j) / d
    # p_i is positions[i_idx], p_j is positions[j_idx]
    diff_vec = positions[i_idx] - positions[j_idx]
    inv_d = pair_weights / d
    # Contribution to grad[i] is +inv_d * diff_vec
    # Contribution to grad[j] is -inv_d * diff_vec
    
    # Reshape for add.at
    contrib_i = inv_d[:, np.newaxis] * diff_vec
    contrib_j = -inv_d[:, np.newaxis] * diff_vec
    
    # Flatten positions for add.at
    # grad is shape (2*N,)
    # positions[i] corresponds to grad[2*i] and grad[2*i+1]
    
    # We need to scatter contrib_i into grad at indices 2*i_idx and 2*i_idx+1
    # And contrib_j into grad at indices 2*j_idx and 2*j_idx+1
    
    # Using np.add.at
    idx_i_x = 2 * i_idx
    idx_i_y = 2 * i_idx + 1
    idx_j_x = 2 * j_idx
    idx_j_y = 2 * j_idx + 1
    
    np.add.at(grad, idx_i_x, contrib_i[:, 0])
    np.add.at(grad, idx_i_y, contrib_i[:, 1])
    np.add.at(grad, idx_j_x, contrib_j[:, 0])
    np.add.at(grad, idx_j_y, contrib_j[:, 1])
    
    return soft_min_val, grad

def _objective(pos_flat, beta):
    """Objective function for minimization (negative soft radius)."""
    positions = pos_flat.reshape(-1, 2)
    val, grad = _soft_min_radius_and_grad(positions, beta)
    # We want to maximize val, so minimize -val
    return -val, -grad

def _generate_hexagonal_grid(n_circles):
    """Generate a hexagonal grid of n_circles points."""
    points = []
    # Approximate side length of square needed for N circles in hex grid
    # N ~ (sqrt(3)/2) * (side/d)^2
    # side ~ d * sqrt(2N/sqrt(3))
    # d is distance between centers.
    # We want to fit in [-0.5, 0.5].
    
    # Let's generate a large grid and crop
    max_rows = 25
    d = 1.0
    for row in range(-max_rows, max_rows + 1):
        offset = (row % 2) * 0.5 * d
        for col in range(-max_rows, max_rows + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance from center
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx].copy()
    
    # Scale to fit in [-0.5, 0.5]
    max_coord = np.max(np.abs(points))
    if max_coord > 0.5:
        points = points * (0.5 / max_coord)
        
    return points

def _optimize_with_lbfgsb(start_pos, beta, time_limit, start_time):
    """Run L-BFGS-B optimization."""
    pos_flat = start_pos.reshape(-1)
    
    def fun_and_grad(x):
        if time.time() > start_time + time_limit:
            raise TimeoutError
        val, grad = _objective(x, beta)
        return val, grad
        
    # We catch TimeoutError to stop early if needed, but minimize might not handle it gracefully.
    # Better to check time in a wrapper or just rely on the maxiter and time check in construct.
    # For now, we'll just run it. The time check in construct will prevent starting new runs.
    
    try:
        res = minimize(
            fun_and_grad,
            pos_flat,
            method='L-BFGS-B',
            jac=True,
            bounds=[(-0.5, 0.5)] * 2 * N,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
        )
        return res.x.reshape(-1, 2)
    except Exception as e:
        # If optimization fails, return the start position
        return start_pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_positions = None
    
    # 1. Hexagonal Grid Init
    hex_pos = _generate_hexagonal_grid(N)
    hex_score = _true_radius(hex_pos)
    if hex_score > best_score:
        best_score = hex_score
        best_positions = hex_pos.copy()
        
    # 2. Square Grid Init
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_pos = grid[:N].copy()
    sq_score = _true_radius(sq_pos)
    if sq_score > best_score:
        best_score = sq_score
        best_positions = sq_pos.copy()
        
    # 3. Optimization from Hex Grid
    if time.time() < deadline - 30:
        try:
            opt_pos = _optimize_with_lbfgsb(hex_pos, beta=50.0, time_limit=30, start_time=start_time)
            opt_score = _true_radius(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
            
    # 4. Optimization from Square Grid
    if time.time() < deadline - 30:
        try:
            opt_pos = _optimize_with_lbfgsb(sq_pos, beta=50.0, time_limit=30, start_time=start_time)
            opt_score = _true_radius(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
            
    # 5. Optimization from Perturbed Hex Grid (Random Restart)
    if time.time() < deadline - 20:
        try:
            rng = np.random.RandomState(42)
            perturbed = hex_pos + rng.normal(0, 0.01, size=hex_pos.shape)
            perturbed = np.clip(perturbed, -0.49, 0.49)
            opt_pos = _optimize_with_lbfgsb(perturbed, beta=50.0, time_limit=20, start_time=start_time)
            opt_score = _true_radius(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
            
    # 6. Final Polish with higher beta (sharper min)
    if time.time() < deadline - 10 and best_positions is not None:
        try:
            # Use the best found so far as start
            opt_pos = _optimize_with_lbfgsb(best_positions, beta=100.0, time_limit=10, start_time=start_time)
            opt_score = _true_radius(opt_pos)
            if opt_score > best_score:
                best_score = opt_score
                best_positions = opt_pos.copy()
        except Exception:
            pass
            
    # Ensure validity
    if best_positions is None:
        best_positions = hex_pos
        
    return best_positions

# EVOLVE-BLOCK-END
