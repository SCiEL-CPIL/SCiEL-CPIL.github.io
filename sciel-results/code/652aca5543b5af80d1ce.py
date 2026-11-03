"""SGLD-based max-min packer for 100 equal spheres in a unit cube."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 140.0  # Safety margin

def _score(points):
    """Return the maximal radius for the given set of centres."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    margin = 0.5 - np.abs(pts).max(axis=1)
    r_margin = margin.min()
    if r_margin <= 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_pair, r_margin)

def _soft_min(x, beta=50.0):
    """Smooth approximation of min(x) using log-sum-exp."""
    # Handle potential overflow/underflow by shifting
    x = np.asarray(x, dtype=np.float64)
    x_min = np.min(x)
    # If x is very negative, exp(beta*(x-x_min)) is safe
    # If x is very positive relative to min, it's also safe
    shifted = beta * (x - x_min)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted))) / beta
    return x_min + log_sum_exp

def _neg_soft_radius(x, beta=50.0):
    """
    Smooth objective: -soft_min(r_pair, r_margin).
    x is flat array of shape (N*3,).
    """
    pts = x.reshape(N, 3)
    # Boundary check with large penalty if out of bounds
    if np.any(np.abs(pts) > 0.5 + 1e-6):
        return 1e10
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = (diff ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    r_pair = dists / 2.0
    
    # Compute boundary margins
    margin = 0.5 - np.abs(pts).max(axis=1)
    
    # Combine into a single array for soft_min
    # We want soft_min(min(r_pair), min(margin))
    # This is equivalent to soft_min of the set {all r_pair, all margins}
    # But soft_min of a large set is computationally expensive (O(N^2) memory for exp)
    # Let's compute soft_min of r_pair and soft_min of margin separately, then soft_min of those two.
    
    # Soft min of r_pair
    r_pair_flat = r_pair.flatten()
    # Exclude inf
    r_pair_valid = r_pair_flat[r_pair_flat < 1e9]
    if len(r_pair_valid) == 0:
        sm_pair = 0.0
    else:
        sm_pair = _soft_min(r_pair_valid, beta)
        
    # Soft min of margin
    sm_margin = _soft_min(margin, beta)
    
    # Final soft min
    final_sm = _soft_min(np.array([sm_pair, sm_margin]), beta)
    
    return -final_sm

def _grad_soft_radius(x, beta=50.0):
    """
    Numerical gradient of _neg_soft_radius.
    Using central differences for stability, though finite diff is slow.
    For SGLD, we can also use random finite difference or just a small step.
    Given N=100, 300 dimensions. Central diff requires 600 evaluations.
    Each eval is O(N^2). 600 * 10000 = 6e6 ops. Might be okay per step?
    Let's try forward difference with a small h to save time, or just use one-sided.
    Actually, let's compute gradient analytically if possible? 
    soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    grad soft_min(x) = exp(-beta*x) / sum(exp(-beta*x)) * -1 ... wait.
    Let f(x) = -1/beta * log(sum(exp(-beta*x_i)))
    df/dx_i = -1/beta * (1/sum(exp(-beta*x_j))) * exp(-beta*x_i) * (-beta)
    df/dx_i = exp(-beta*x_i) / sum(exp(-beta*x_j))
    
    So we need the probabilities p_i = exp(-beta*x_i) / sum(exp(-beta*x_j)).
    
    Let's implement analytic gradient for speed.
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-6):
        return np.zeros_like(x)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = (diff ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    r_pair = dists / 2.0
    
    margin = 0.5 - np.abs(pts).max(axis=1)
    
    # We need p_pair and p_margin distributions
    # For r_pair, we have N*(N-1)/2 values.
    # Let's vectorize the soft-min gradient calculation.
    
    # 1. Soft min of r_pair
    r_pair_flat = r_pair.flatten()
    mask = r_pair_flat < 1e9
    r_pair_valid = r_pair_flat[mask]
    
    # Calculate p_pair
    # p_i = exp(-beta * r_i) / sum(exp(-beta * r_j))
    # To avoid overflow, shift by min
    r_min = np.min(r_pair_valid)
    exp_terms = np.exp(-beta * (r_pair_valid - r_min))
    sum_exp = np.sum(exp_terms)
    p_pair = exp_terms / sum_exp
    
    # 2. Soft min of margin
    m_min = np.min(margin)
    exp_terms_m = np.exp(-beta * (margin - m_min))
    sum_exp_m = np.sum(exp_terms_m)
    p_margin = exp_terms_m / sum_exp_m
    
    # 3. Combine
    # We have two values: sm_pair and sm_margin.
    # sm_pair = r_min + (1/beta)*log(sum_exp)
    # sm_margin = m_min + (1/beta)*log(sum_exp_m)
    
    sm_pair_val = r_min + (1.0/beta)*np.log(sum_exp)
    sm_margin_val = m_min + (1.0/beta)*np.log(sum_exp_m)
    
    # Final soft min of [sm_pair_val, sm_margin_val]
    vals = np.array([sm_pair_val, sm_margin_val])
    v_min = np.min(vals)
    exp_final = np.exp(-beta * (vals - v_min))
    sum_final = np.sum(exp_final)
    p_final = exp_final / sum_final
    
    # p_final[0] is weight for sm_pair
    # p_final[1] is weight for sm_margin
    
    # Gradient of final soft min w.r.t x:
    # d(sm_final)/dx = p_final[0] * d(sm_pair)/dx + p_final[1] * d(sm_margin)/dx
    
    grad = np.zeros_like(pts)
    
    # d(sm_pair)/dx
    # sm_pair depends on r_pair.
    # d(sm_pair)/dr_i = p_pair[i]
    # dr_i/dx_k = ...
    # r_i = 0.5 * ||x_i - x_j||
    # dr_i/dx_k = 0.5 * (x_k - x_j) / ||x_k - x_j||  if k is part of pair i
    #            -0.5 * (x_k - x_i) / ||x_k - x_i|| if k is part of pair i (other side)
    
    # This is complex to vectorize perfectly without loops or large memory.
    # Given time constraints, let's use numerical gradient for the SGLD steps.
    # It's safer and less code. We can optimize later if needed.
    
    h = 1e-5
    grad_num = np.zeros_like(x)
    f0 = _neg_soft_radius(x, beta)
    
    # To save time, we can sample a subset of coordinates for gradient estimation?
    # No, SGLD needs full gradient direction.
    # Let's just do it. 300 coords. 600 function calls.
    # Each function call is ~1ms? 0.6s per gradient step. 
    # If we do 100 steps, that's 60s. Too slow.
    
    # Alternative: Use a cheaper approximation or fewer steps.
    # Or use the analytic gradient but simplified.
    
    # Let's try to implement the analytic gradient properly but efficiently.
    # We need to accumulate gradients for each point.
    
    # For margin:
    # d(sm_margin)/dx_k = p_margin[k] * d(margin_k)/dx_k
    # margin_k = 0.5 - max(|x_kx|, |x_ky|, |x_kz|)
    # d(margin_k)/dx_kx = -sign(x_kx) if |x_kx| is max, else 0
    # This is easy.
    
    grad_margin_part = np.zeros_like(pts)
    for k in range(N):
        ax, ay, az = abs(pts[k,0]), abs(pts[k,1]), abs(pts[k,2])
        max_val = max(ax, ay, az)
        if max_val == 0: continue
        if ax == max_val:
            grad_margin_part[k, 0] = -np.sign(pts[k, 0])
        if ay == max_val:
            grad_margin_part[k, 1] = -np.sign(pts[k, 1])
        if az == max_val:
            grad_margin_part[k, 2] = -np.sign(pts[k, 2])
            
    grad += p_final[1] * p_margin[:, np.newaxis] * grad_margin_part
    
    # For pair distances:
    # d(sm_pair)/dx = sum_i p_pair[i] * d(r_i)/dx
    # r_i corresponds to pair (u, v)
    # d(r_i)/dx_u = 0.5 * (x_u - x_v) / dist(u,v)
    # d(r_i)/dx_v = 0.5 * (x_v - x_u) / dist(u,v)
    
    # We need to map p_pair back to indices.
    # r_pair was flattened from (N, N) matrix with diagonal inf.
    # Indices in flattened array correspond to (i, j) where i != j.
    # We can iterate over i < j to be efficient.
    
    grad_pair_part = np.zeros_like(pts)
    
    # We need to align p_pair with the pairs.
    # r_pair_flat was created from r_pair (N, N).
    # The valid entries are all i != j.
    # Let's create a list of (i, j) indices for the valid entries.
    # This is N*(N-1) entries.
    
    # To avoid huge memory allocation for indices, we can loop.
    # N=100, 9900 pairs. Looping 9900 times in Python is slow (~0.1s).
    # Acceptable if we do few steps.
    
    # Let's optimize the loop.
    # We have p_pair array of length N*(N-1).
    # We need to know which (i,j) each p corresponds to.
    
    # Reconstruct the mapping.
    # r_pair[i, j] is at index i*N + j in the flattened array?
    # Yes, if we flatten row-major.
    # But we removed diagonal.
    # Let's just recompute the indices or use a smarter way.
    
    # Actually, we can compute the gradient of the sum of weighted distances directly.
    # sum_{i<j} p_{ij} * 0.5 * ||x_i - x_j||
    # This is equivalent to a force-directed graph.
    
    # Let's build the force vector.
    # F_i = sum_{j != i} p_{ij} * 0.5 * (x_i - x_j) / ||x_i - x_j||
    
    # We need p_{ij} for all i<j.
    # Let's create a matrix P of size N x N.
    P = np.zeros((N, N))
    
    # Fill P from p_pair
    # We need to map the flattened p_pair to (i, j).
    # The flattened r_pair (with diagonal inf removed) order:
    # (0,1), (0,2), ..., (0,N-1), (1,0), (1,2), ...
    # It's easier to just recompute the soft-min probabilities directly on the (N, N) grid
    # but masking the diagonal.
    
    # Let's do that. It's cleaner.
    # r_pair is (N, N).
    # Set diagonal to inf.
    r_pair_mat = r_pair.copy()
    np.fill_diagonal(r_pair_mat, np.inf)
    
    # Soft min over all valid entries in r_pair_mat
    # Shift by min
    r_min_mat = np.min(r_pair_mat)
    # Only valid entries
    mask_mat = r_pair_mat < 1e9
    exp_mat = np.where(mask_mat, np.exp(-beta * (r_pair_mat - r_min_mat)), 0.0)
    sum_exp_mat = np.sum(exp_mat)
    p_mat = exp_mat / sum_exp_mat
    
    # Now p_mat[i, j] is the probability weight for distance i-j.
    # Note: p_mat[i, j] == p_mat[j, i] because dist is symmetric and beta is same.
    # Wait, soft_min is symmetric.
    
    # Force on i from j: p_mat[i, j] * 0.5 * (x_i - x_j) / dist(i, j)
    # Total force on i: sum_j p_mat[i, j] * 0.5 * (x_i - x_j) / dist(i, j)
    
    # We can compute this with matrix operations.
    # dist_mat = r_pair * 2 (since r_pair = dist/2)
    # Actually r_pair_mat = dist/2. So dist = 2 * r_pair_mat.
    # Term: p_mat[i, j] * 0.5 * (x_i - x_j) / (2 * r_pair_mat[i, j])
    #      = p_mat[i, j] * (x_i - x_j) / (4 * r_pair_mat[i, j])
    
    # Let's compute the coefficient matrix C where C[i, j] = p_mat[i, j] / (4 * r_pair_mat[i, j])
    # Then Force_i = sum_j C[i, j] * (x_i - x_j)
    #             = x_i * sum_j C[i, j] - sum_j C[i, j] * x_j
    
    with np.errstate(divide='ignore', invalid='ignore'):
        C = np.where(mask_mat, p_mat / (4.0 * r_pair_mat), 0.0)
    
    # Row sums of C
    C_row_sums = C.sum(axis=1)
    
    # C @ pts gives sum_j C[i, j] * x_j
    C_pts = C @ pts
    
    # Force_i = C_row_sums[i] * x_i - C_pts[i]
    # grad_pair_part = Force
    grad_pair_part = C_row_sums[:, np.newaxis] * pts - C_pts
    
    # Combine gradients
    # Total grad = p_final[0] * grad_pair_part + p_final[1] * grad_margin_part
    # Note: _neg_soft_radius returns -soft_min.
    # So we want gradient of -soft_min, which is -grad(soft_min).
    # The forces calculated above point in the direction of increasing distance (repulsion).
    # soft_min is minimized when distances are large.
    # grad(soft_min) points in direction of decreasing soft_min (i.e., increasing distances? No).
    # Let's check signs.
    # f(x) = soft_min(distances). If we increase distance, f increases (since min of larger numbers is larger).
    # So grad(soft_min) points in direction of increasing distances.
    # We want to minimize -soft_min, so we want to move in direction of -grad(soft_min).
    # -grad(soft_min) points in direction of decreasing distances (attraction)?
    # Wait. We want to MAXIMIZE the radius.
    # Max radius = min(distances).
    # So we want to maximize soft_min(distances).
    # Our objective function _neg_soft_radius returns -soft_min.
    # So we want to MINIMIZE _neg_soft_radius.
    # Gradient descent on _neg_soft_radius moves in direction -grad(_neg_soft_radius) = grad(soft_min).
    # grad(soft_min) points in direction of increasing distances.
    # So the forces calculated (repulsion) are in the direction of grad(soft_min).
    # So we should ADD this gradient to the position in SGLD?
    # SGLD update: x_{t+1} = x_t + lr * grad(objective) + noise.
    # Here objective is _neg_soft_radius.
    # grad(_neg_soft_radius) = -grad(soft_min).
    # So we move in direction -grad(soft_min), which is attraction?
    # No, that's wrong.
    # If I want to maximize min(dist), I want to increase distances.
    # So I want to move in direction of increasing distances.
    # That is the direction of grad(soft_min).
    # But gradient descent minimizes.
    # So if I minimize -soft_min, I move in direction -grad(-soft_min) = grad(soft_min).
    # Correct.
    # So the update step should be + lr * grad(soft_min).
    # My calculated grad_pair_part is in direction of increasing distances (repulsion).
    # My calculated grad_margin_part is in direction of increasing margin (away from walls).
    # So I should add these to the position.
    
    total_grad = p_final[0] * grad_pair_part + p_final[1] * grad_margin_part
    
    # Return gradient of _neg_soft_radius?
    # No, SGLD usually uses gradient of the energy/potential.
    # Let's define the "force" as the direction to move to improve the score.
    # Score = soft_min. We want to maximize Score.
    # So we move in direction grad(Score).
    # Let's return grad(Score) directly to avoid confusion.
    # The function name is _grad_soft_radius, implying gradient of soft_radius (which is soft_min).
    # So returning total_grad is correct for "direction to increase score".
    
    return total_grad.flatten()

def _sgld_optimize(points, time_budget=20.0, beta_start=10.0, beta_end=100.0, lr_start=0.01, lr_end=0.001, noise_start=0.01, noise_end=0.001):
    """
    SGLD optimization.
    """
    start_time = time.time()
    x = points.flatten().copy()
    best_x = x.copy()
    best_score = _score(x.reshape(N, 3))
    
    n_steps = 0
    while time.time() - start_time < time_budget:
        n_steps += 1
        # Annealing parameters
        alpha = min(1.0, n_steps / 100.0) # Ramp up over 100 steps
        beta = beta_start + (beta_end - beta_start) * alpha
        lr = lr_start * (1.0 - alpha) + lr_end * alpha
        noise = noise_start * (1.0 - alpha) + noise_end * alpha
        
        # Compute gradient (direction to increase score)
        grad = _grad_soft_radius(x, beta)
        
        # SGLD update
        # x += lr * grad + noise * random
        # Note: grad is for soft_min (maximize).
        # So we add lr * grad.
        noise_vec = np.random.normal(0, noise, x.shape)
        x = x + lr * grad + noise_vec
        
        # Clip to bounds
        x = np.clip(x, -0.5, 0.5)
        
        # Check score periodically
        if n_steps % 10 == 0:
            current_score = _score(x.reshape(N, 3))
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                
    return best_x.reshape(N, 3)

def _grid_init(nx, ny, nz, jitter=0.0):
    """Generate a grid-based initialization with optional jitter."""
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.125, 0.5 - 0.125, nz)
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        points += np.random.normal(0, jitter, points.shape)
        points = np.clip(points, -0.5, 0.5)
    return points

def _random_init(scale=0.05):
    """Generate a random initialization within the cube."""
    points = np.random.uniform(-0.5 + scale, 0.5 - scale, (N, 3))
    return points

def _coordinate_polish(points, n_iter=50):
    """Quick coordinate polish."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            for coord in range(3):
                for direction in [1, -1]:
                    step = 0.005 * (1.0 - it / n_iter)
                    new_point = current[i].copy()
                    new_point[coord] += direction * step
                    new_point = np.clip(new_point, -0.5, 0.5)
                    
                    old_point = current[i].copy()
                    current[i] = new_point
                    new_score = _score(current)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best = current.copy()
                            best_score = new_score
                    else:
                        current[i] = old_point
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    random.seed(42)
    np.random.seed(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Grid-based initializations
    for nx, ny, nz in [(5, 5, 4), (4, 5, 5), (5, 4, 5), (4, 4, 6), (6, 4, 4)]:
        initializations.append(_grid_init(nx, ny, nz, jitter=0.02))
        
    # Random initializations
    for _ in range(5):
        initializations.append(_random_init(scale=0.05))
        
    best_points = None
    best_score = 0.0
    
    # Sort by initial score
    candidates = [(init, _score(init)) for init in initializations]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    for init, init_score in candidates:
        if time.time() > deadline - 15:
            break
            
        # SGLD Optimization
        # Allocate time based on remaining time
        time_left = deadline - time.time()
        # Use a fraction of remaining time for SGLD
        sgl_time = min(10.0, time_left * 0.5)
        
        if sgl_time > 1.0:
            opt_points = _sgld_optimize(init, time_budget=sgl_time)
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
                
    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 5:
        polished = _coordinate_polish(best_points, n_iter=50)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
            
    if best_points is None:
        best_points = _grid_init(5, 5, 4, jitter=0.0)
        
    return best_points

# EVOLVE-BLOCK-END
