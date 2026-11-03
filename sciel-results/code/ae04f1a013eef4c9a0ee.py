"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^3."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centres), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    n = len(centres)
    if n < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _surrogate_objective(x, beta=500.0):
    """
    Smooth surrogate objective: -log-sum-exp(beta * (r_max - constraints))
    We want to maximize min(constraints).
    Let constraints c_i be distances (pair/2 and wall).
    We maximize min(c_i).
    Equivalent to minimizing -min(c_i).
    Smooth approximation: - (1/beta) * log(sum(exp(beta * c_i))) is NOT right for max-min.
    
    Let's use: f(x) = - (1/beta) * log( sum_{i} exp(-beta * c_i(x)) )
    As beta -> inf, this approaches max_i(-c_i) = -min_i(c_i).
    So minimizing this function maximizes the minimum constraint.
    
    Constraints:
    1. Pairwise: d_ij / 2 for all i < j
    2. Wall: 0.5 - max(|x_i|, |y_i|, |z_i|) for all i
    
    We compute the negative log-sum-exp of the negative constraints.
    """
    centres = x.reshape(-1, 3)
    n = len(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # We need distances, but for LSE we can work with squared distances?
    # No, the objective is based on linear distances (radius).
    # However, sqrt is expensive. 
    # Let's stick to linear distances for correctness of the surrogate.
    dists = np.sqrt(dists_sq)
    pair_constraints = dists / 2.0
    # Flatten upper triangle
    iu = np.triu_indices(n, 1)
    pair_vals = pair_constraints[iu]
    
    # Wall distances
    wall_vals = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Combine all constraints
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Scale
    scaled = -beta * all_vals
    
    # Log-sum-exp trick for stability
    max_scaled = np.max(scaled)
    if not np.isfinite(max_scaled):
        return 0.0
        
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    
    return lse / beta

def _surrogate_gradient(x, beta=500.0):
    """
    Gradient of the surrogate objective.
    f(x) = (1/beta) * log( sum_j exp(-beta * c_j(x)) )
    df/dx = (1/beta) * (1/sum) * sum_j (-beta * exp(-beta * c_j) * dc_j/dx)
    df/dx = - (sum_j w_j * dc_j/dx) where w_j = exp(-beta*c_j) / sum_k exp(-beta*c_k)
    
    Note: The objective to MINIMIZE is -min(c).
    My previous definition was f = (1/beta) log(sum exp(-beta c)).
    If c is large (good), -beta c is very negative, exp is small.
    If c is small (bad), -beta c is less negative, exp is larger.
    The term with smallest c dominates the sum.
    So f approximates -c_min.
    We want to minimize f, which means maximizing c_min. Correct.
    
    Gradient of f wrt x:
    f = (1/beta) * log(S) where S = sum_j e_j, e_j = exp(-beta c_j)
    df/dx = (1/beta) * (1/S) * sum_j (de_j/dx)
    de_j/dx = e_j * (-beta) * dc_j/dx
    df/dx = (1/beta) * (1/S) * sum_j (-beta e_j dc_j/dx)
    df/dx = - sum_j (e_j/S) * dc_j/dx
    df/dx = - sum_j w_j * dc_j/dx
    where w_j = e_j / S.
    
    So we compute weighted sum of gradients of constraints, with weights favoring the smallest constraints.
    """
    centres = x.reshape(-1, 3)
    n = len(centres)
    grad = np.zeros_like(x)
    
    # Pairwise constraints
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # c_ij = d_ij / 2
    # dc_ij/dx_i = (x_i - x_j) / d_ij
    # dc_ij/dx_j = (x_j - x_i) / d_ij = -dc_ij/dx_i
    
    # We need weights w_ij = exp(-beta * c_ij) / S
    # To avoid underflow/overflow, we compute unnormalized weights and normalize.
    # But S includes wall terms too.
    
    # Let's compute all unnormalized weights first
    # Pair weights
    pair_c = dists / 2.0
    np.fill_diagonal(pair_c, np.inf) # Avoid self
    # Only upper triangle matters for sum, but gradient couples i and j.
    # Let's iterate over pairs to save memory if possible, but N=50 is small (1225 pairs).
    
    # Compute raw weights
    # We need to be careful with inf.
    # Set diagonal to 0 for weight calculation? No, diagonal is inf distance, so weight 0.
    # Replace inf with large number for exp? Or just mask.
    
    # Efficiently:
    # 1. Compute pair distances and gradients
    # 2. Compute wall distances and gradients
    # 3. Combine
    
    # Pair part
    # diff[i,j] = x_i - x_j
    # d_ij = ||diff[i,j]||
    # grad_c_ij_i = diff[i,j] / d_ij
    # grad_c_ij_j = -diff[i,j] / d_ij
    
    # We want sum_j w_j * dc_j/dx
    # For pairs, contribution to x_i from pair (i,j) is w_ij * (x_i - x_j)/d_ij
    # Contribution to x_j from pair (i,j) is w_ij * (x_j - x_i)/d_ij
    
    # Let's compute the sum of weights * gradients.
    # w_ij = exp(-beta * d_ij/2)
    
    # To handle scaling, let's find the max constraint value to stabilize exp?
    # Actually, weights sum to 1 after division by S.
    # S = sum_pairs exp(-beta d/2) + sum_walls exp(-beta w)
    
    # Let's compute log_weights to avoid underflow if beta is large.
    # But beta=500, d~0.1, exp(-25) is tiny. exp(-0) is 1.
    # The largest term (smallest distance) will dominate.
    # We can just compute exp and let small ones vanish.
    
    pair_weights = np.zeros((n, n))
    # Mask diagonal
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    # Only compute for valid pairs
    valid_dists = dists[mask]
    # d is > 0
    vals = -beta * (valid_dists / 2.0)
    # Clip to avoid overflow in exp? vals are negative, so exp < 1. No overflow.
    # But if d is very small, vals close to 0.
    weights_raw = np.exp(vals)
    
    # Put back into matrix
    pair_weights[mask] = weights_raw
    # Symmetric, so pair_weights[j,i] = pair_weights[i,j]
    # Wait, my mask is upper/lower? np.triu_indices? 
    # I used full mask excluding diagonal. So I computed all pairs twice.
    # That's fine, weights are symmetric.
    
    # Wall part
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    wall_vals = -beta * wall_dists
    wall_weights = np.exp(wall_vals)
    
    # Total Sum S
    S = np.sum(pair_weights) + np.sum(wall_weights)
    
    if S == 0 or not np.isfinite(S):
        return grad # Should not happen
        
    # Normalize weights
    pair_weights /= S
    wall_weights /= S
    
    # Now compute gradient
    # Pair contribution:
    # grad_x_i += sum_{j != i} w_ij * (x_i - x_j) / d_ij
    # We have diff[i,j] = x_i - x_j
    # term = diff * (pair_weights / dists)[:, :, np.newaxis]
    # Note: dists has 0 on diagonal, but pair_weights has 0 on diagonal.
    # To avoid 0/0, set dists diagonal to 1.
    safe_dists = dists.copy()
    np.fill_diagonal(safe_dists, 1.0)
    
    grad_pair = np.sum(diff * (pair_weights / safe_dists)[:, :, np.newaxis], axis=1)
    
    # Wall contribution:
    # c_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    # dc_i/dx_i = -sign(x_i) if x_i is the max abs component
    # If multiple are max, subgradient. We pick the one with max abs.
    # Let's find the index of max abs for each sphere.
    abs_centres = np.abs(centres)
    max_idx = np.argmax(abs_centres, axis=1)
    
    # grad_wall_i = -w_i * sign(x_i[max_idx]) in the max_idx dimension
    # w_i is wall_weights[i]
    
    grad_wall = np.zeros_like(centres)
    # For each sphere i:
    # grad_wall[i, max_idx[i]] = -wall_weights[i] * sign(centres[i, max_idx[i]])
    
    # Vectorized:
    rows = np.arange(n)
    grad_wall[rows, max_idx] = -wall_weights * np.sign(centres[rows, max_idx])
    
    # Total gradient of f (which we minimize)
    # df/dx = - (sum w dc/dx)
    # So we return - (grad_pair + grad_wall)
    
    return -(grad_pair + grad_wall)

def _optimize_lbgfgs(centres, time_limit=10.0):
    """
    Run L-BFGS-B on the surrogate objective.
    """
    start_time = time.time()
    x0 = centres.flatten()
    
    # Bounds: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use L-BFGS-B
    # maxiter and ftol to stop early if converged
    res = minimize(
        _surrogate_objective,
        x0,
        jac=_surrogate_gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    # Check time
    if time.time() - start_time > time_limit:
        # If it took too long, we might have a better result in res.x anyway,
        # but we should check if it finished.
        pass
        
    x_opt = res.x.reshape(-1, 3)
    # Ensure validity
    x_opt = np.clip(x_opt, -0.5, 0.5)
    
    return x_opt

def _generate_grid_start():
    """Generate a 4x4x4 grid start, taking first 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle to ensure we don't just take a corner
    np.random.shuffle(grid)
    return grid[:N].copy()

def _generate_random_start():
    """Generate random points."""
    return np.random.rand(N, 3) - 0.5

def construct():
    """Construct 50 sphere centres in [-0.5, 0.5]^3 to maximize minimum radius."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_centres = None
    best_radius = 0.0
    
    np.random.seed(42)
    
    # Multi-start
    # 1. Grid start
    # 2. Random starts
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if i == 0:
            initial = _generate_grid_start()
        else:
            initial = _generate_random_start()
            # Quick pre-spread to avoid initial overlaps causing numerical issues in gradient?
            # L-BFGS handles overlaps, but starting far apart is better.
            # Let's do a few steps of simple repulsion or just rely on L-BFGS.
            # L-BFGS on log-sum-exp is robust to overlaps.
            
        # Allocate time for this optimization
        # If we have 100s left and 10 restarts, 10s each.
        time_remaining = deadline - time.time() - 5.0
        time_per_restart = time_remaining / max(1, num_restarts - i)
        
        opt_centres = _optimize_lbgfgs(initial, time_limit=time_per_restart)
        
        # Evaluate true radius
        radius = _compute_radius(opt_centres)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = opt_centres.copy()
            
        # Optional: refine the best one further with higher precision?
        # L-BFGS is usually good.
        
    # Final polish: run L-BFGS on the best found with more iterations/time
    if best_centres is not None and time.time() < deadline - 10.0:
        time_remaining = deadline - time.time() - 2.0
        refined = _optimize_lbgfgs(best_centres, time_limit=time_remaining)
        refined_radius = _compute_radius(refined)
        if refined_radius > best_radius:
            best_radius = refined_radius
            best_centres = refined
            
    return best_centres
# EVOLVE-BLOCK-END
