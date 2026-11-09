"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(positions):
    """Compute the minimum radius given positions in [-0.5, 0.5]^2."""
    n = len(positions)
    if n == 0:
        return 0.0
    # Distance to boundary
    min_boundary = np.min(np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ))
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_boundary, min_pair / 2.0)


def _soft_min_radius(positions, beta=50.0):
    """
    Smooth surrogate for the min radius using log-sum-exp.
    Returns a smooth approximation of min(boundary_dist, min_pair_dist/2).
    Higher beta = sharper approximation to true min.
    """
    n = len(positions)
    # Boundary distances
    boundary = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_radii = dist / 2.0
    
    # Combine all constraints into a single array
    all_radii = np.concatenate([boundary, pair_radii.ravel()])
    
    # Soft-min via log-sum-exp: -logsumexp(-beta * x) / beta
    # This approximates min(x) as beta -> inf
    neg_beta_x = -beta * all_radii
    max_val = np.max(neg_beta_x)
    if not np.isfinite(max_val):
        return 0.0
    log_sum = np.log(np.sum(np.exp(neg_beta_x - max_val))) + max_val
    soft_min = -log_sum / beta
    
    return soft_min


def _soft_min_radius_grad(positions, beta=50.0):
    """
    Compute gradient of soft-min radius w.r.t. positions.
    Returns (value, gradient_array_of_shape_(2N,))
    """
    n = len(positions)
    pos = positions.reshape(-1)
    
    # Boundary distances
    boundary = np.minimum(
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    )
    
    # Pairwise distances / 2
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair_radii = dist / 2.0
    
    # Combine
    all_radii = np.concatenate([boundary, pair_radii.ravel()])
    
    # Soft-min computation
    neg_beta_x = -beta * all_radii
    max_val = np.max(neg_beta_x)
    if not np.isfinite(max_val):
        return 0.0, np.zeros(2 * n)
    
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    log_sum = np.log(sum_exp) + max_val
    soft_min = -log_sum / beta
    
    # Weights for each constraint in the soft-min
    weights = exp_vals / sum_exp  # These sum to 1
    
    # Gradient w.r.t. positions
    grad = np.zeros(2 * n)
    
    # Boundary gradients
    # boundary_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # d(0.5 - |x_i|)/dx_i = -sign(x_i)
    # d(0.5 - |y_i|)/dy_i = -sign(y_i)
    for i in range(n):
        w_bx = weights[i]  # weight for boundary constraint i
        # But we need to be careful: boundary[i] is the min of two terms
        # For simplicity, approximate gradient as weighted average
        dx_sign = np.sign(positions[i, 0]) if positions[i, 0] != 0 else 0
        dy_sign = np.sign(positions[i, 1]) if positions[i, 1] != 0 else 0
        grad[2*i] += w_bx * (-dx_sign)
        grad[2*i+1] += w_bx * (-dy_sign)
    
    # Pairwise gradients
    # For pair (i,j), radius = dist(i,j)/2
    # d(dist/2)/dx_i = (x_i - x_j) / dist
    # d(dist/2)/dx_j = (x_j - x_i) / dist
    for i in range(n):
        for j in range(i+1, n):
            # Find index in pair_radii.ravel()
            # pair_radii is upper triangular, flattened
            # Index for (i,j) where i<j: in the flattened upper triangle
            # The flat index for position (i,j) in a (n,n) matrix with diagonal filled
            # We'll just recompute
            d = dist[i, j]
            if d < 1e-15:
                continue
            w_pair = weights[n + i*n + j]  # weight for this pair
            # Gradient contribution
            dir_x = (positions[i, 0] - positions[j, 0]) / d
            dir_y = (positions[i, 1] - positions[j, 1]) / d
            grad[2*i] += w_pair * dir_x
            grad[2*i+1] += w_pair * dir_y
            grad[2*j] += w_pair * (-dir_x)
            grad[2*j+1] += w_pair * (-dir_y)
    
    return soft_min, grad


def _generate_hexagonal_grid(n_circles):
    """Generate a hexagonal grid of n_circles points centered in [-0.5, 0.5]^2."""
    points = []
    max_rows = int(np.ceil(np.sqrt(n_circles * 2 / np.sqrt(3)))) + 2
    d = 1.0
    for row in range(-max_rows, max_rows + 1):
        offset = (row % 2) * 0.5 * d
        for col in range(-max_rows, max_rows + 1):
            x = col * d + offset
            y = row * np.sqrt(3) / 2 * d
            points.append([x, y])
    
    points = np.array(points)
    dists = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(dists)[:n_circles]
    points = points[idx].copy()
    
    max_coord = np.max(np.abs(points))
    if max_coord > 0.5:
        points = points * (0.5 / max_coord)
    
    return points


def _gradient_ascent_optimize(positions, max_iters=5000, lr_init=0.001, beta_init=10.0, beta_final=100.0):
    """
    Gradient ascent on soft-min radius with adaptive learning rate.
    """
    positions = positions.copy().reshape(-1, 2)
    n = len(positions)
    
    best_val = _score(positions)
    best_pos = positions.copy()
    
    for it in range(max_iters):
        # Adaptive beta: start low (smooth), increase toward sharp
        progress = it / max_iters
        beta = beta_init + (beta_final - beta_init) * (progress ** 0.5)
        
        # Adaptive learning rate: decrease over time
        lr = lr_init * (0.999 ** it)
        lr = max(lr, 1e-6)
        
        val, grad = _soft_min_radius_grad(positions, beta)
        
        if not np.isfinite(val):
            break
        
        # Update
        positions_new = positions + lr * grad.reshape(-1, 2)
        
        # Clamp to bounds
        positions_new = np.clip(positions_new, -0.5, 0.5)
        
        # Check improvement
        new_score = _score(positions_new)
        if new_score > best_val:
            best_val = new_score
            best_pos = positions_new.copy()
        
        positions = positions_new
        
        # Early stopping if no improvement for a while
        if it > 100 and it % 100 == 0:
            if abs(_score(positions) - best_val) < 1e-12:
                break
    
    return best_pos, best_val


def _local_refine(positions, max_iters=2000, lr_init=0.0005, beta=50.0):
    """
    Local refinement with fixed beta and decreasing learning rate.
    """
    positions = positions.copy().reshape(-1, 2)
    n = len(positions)
    
    best_val = _score(positions)
    best_pos = positions.copy()
    
    for it in range(max_iters):
        lr = lr_init * (0.9995 ** it)
        lr = max(lr, 1e-8)
        
        val, grad = _soft_min_radius_grad(positions, beta)
        
        if not np.isfinite(val):
            break
        
        positions_new = positions + lr * grad.reshape(-1, 2)
        positions_new = np.clip(positions_new, -0.5, 0.5)
        
        new_score = _score(positions_new)
        if new_score > best_val:
            best_val = new_score
            best_pos = positions_new.copy()
        
        positions = positions_new
    
    return best_pos, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_positions = None
    
    # Generate hexagonal grid initialization
    hex_pos = _generate_hexagonal_grid(N)
    curr_score = _score(hex_pos)
    if curr_score > best_score:
        best_score = curr_score
        best_positions = hex_pos.copy()
    
    # Also try square grid as backup
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_pos = grid[:N].copy()
    sq_score = _score(sq_pos)
    if sq_score > best_score:
        best_score = sq_score
        best_positions = sq_pos.copy()
    
    # Gradient ascent from hexagonal grid
    if time.time() < deadline - 60:
        try:
            refined, ref_score = _gradient_ascent_optimize(
                hex_pos, max_iters=3000, lr_init=0.001, beta_init=10.0, beta_final=80.0
            )
            if ref_score > best_score:
                best_score = ref_score
                best_positions = refined.copy()
        except Exception:
            pass
    
    # Local refinement from best position
    if time.time() < deadline - 30:
        try:
            refined2, ref_score2 = _local_refine(
                best_positions, max_iters=2000, lr_init=0.0005, beta=60.0
            )
            if ref_score2 > best_score:
                best_score = ref_score2
                best_positions = refined2.copy()
        except Exception:
            pass
    
    # Second gradient ascent with different parameters (different restart)
    if time.time() < deadline - 20:
        try:
            # Perturb the best position slightly for a different restart
            rng = np.random.RandomState(42)
            perturbed = best_positions + rng.normal(0, 0.005, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)
            
            refined3, ref_score3 = _gradient_ascent_optimize(
                perturbed, max_iters=2000, lr_init=0.0008, beta_init=20.0, beta_final=80.0
            )
            if ref_score3 > best_score:
                best_score = ref_score3
                best_positions = refined3.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            refined4, ref_score4 = _local_refine(
                best_positions, max_iters=1500, lr_init=0.0003, beta=80.0
            )
            if ref_score4 > best_score:
                best_score = ref_score4
                best_positions = refined4.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_positions is None:
        best_positions = hex_pos
    
    return best_positions

# EVOLVE-BLOCK-END
