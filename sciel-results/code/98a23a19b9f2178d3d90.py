import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    """
    Compute the radius of the largest equal circles that can be placed 
    at the given centers in the square [-0.5, 0.5]^2.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    # Distance to boundaries
    # Centers must be in [-0.5, 0.5]
    # If a center is outside, radius is effectively 0 or negative (invalid)
    # The problem states: "A centre on or outside the square... makes the candidate invalid."
    # However, compute_radius is used for scoring valid candidates. 
    # The evaluator checks validity separately. 
    # For our internal optimization, we want to push centers inside.
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    # If any center is outside, min_boundary_dist will be negative.
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    return r

def soft_min_objective(centers_flat, beta=100.0):
    """
    Objective: -soft_min(constraints)
    Constraints are:
    1. Pairwise distances d_ij
    2. Boundary distances b_ik
    
    We want to maximize min(constraints).
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_k)))
    This is a smooth approximation of min(x_k).
    
    We minimize -soft_min(constraints).
    """
    centers = centers_flat.reshape(N, 2)
    
    # 1. Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Add small epsilon to avoid log(0) or sqrt(0) issues in gradients if needed, 
    # but soft-min handles 0s fine if we are careful.
    # Actually, we need the distances themselves.
    dist = np.sqrt(dist_sq + 1e-12)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # 2. Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    b_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Combine all constraints into a single vector
    # Note: We are minimizing the negative of the soft-min.
    # soft_min = -1/beta * log(sum(exp(-beta * c)))
    # We want to maximize soft_min, so minimize -soft_min.
    # -soft_min = 1/beta * log(sum(exp(-beta * c)))
    
    # To avoid overflow in exp(-beta * c), we use the log-sum-exp trick:
    # log(sum(exp(x_i))) = m + log(sum(exp(x_i - m))) where m = max(x_i)
    # Here x_i = -beta * c_i.
    # max(-beta * c_i) = -beta * min(c_i).
    
    all_constr = np.concatenate([pair_dists, b_dists])
    
    # If any constraint is very large positive, exp(-beta*c) is small.
    # If any constraint is negative (violation), exp(-beta*c) is large.
    
    # Let's compute log-sum-exp of -beta * all_constr
    x = -beta * all_constr
    m = np.max(x)
    if m > 500: # Overflow protection
        # This means min(all_constr) is very negative (large violation)
        # The objective should be very positive (bad)
        # Gradient will be huge pushing away from violation.
        # We can just return a large penalty and zero grad or approx grad.
        # But let's try to be more stable.
        # Actually, if m > 500, exp(x-m) is safe.
        pass
        
    lse = m + np.log(np.sum(np.exp(x - m)))
    
    val = lse / beta
    
    # Gradient calculation
    # d/dc (lse/beta) = (1/beta) * d/dc (m + log(sum(exp(x-m))))
    # d/dc lse = sum( (exp(x_i - m) / sum(exp(x_i - m))) * (dx_i/dc) )
    # dx_i/dc = -beta * dc_i/dc
    
    # We need gradients of constraints wrt centers.
    # This is tricky to vectorize fully with the log-sum-exp weights.
    # Let's compute weights w_i = exp(x_i - m) / sum(exp(x_i - m))
    # Then grad_obj = (1/beta) * sum_i w_i * (-beta * grad_constraint_i)
    #                = - sum_i w_i * grad_constraint_i
    
    weights = np.exp(x - m)
    weights /= np.sum(weights)
    
    grad_centers = np.zeros_like(centers)
    
    # 1. Pairwise gradients
    # For each pair (i, j) in upper triangle:
    # Constraint c_ij = dist_ij
    # grad c_ij wrt c_i = (c_i - c_j) / dist_ij
    # grad c_ij wrt c_j = (c_j - c_i) / dist_ij
    
    # We have weights for pair_dists.
    # pair_dists corresponds to iu indices.
    w_pairs = weights[:len(pair_dists)]
    
    # Vectorized gradient accumulation for pairs
    # diff shape (N, N, 2)
    # dist shape (N, N)
    # We only care about upper triangle.
    
    # Create a matrix of weights for pairs, 0 on diagonal and lower triangle
    w_mat = np.zeros((N, N))
    w_mat[iu] = w_pairs
    
    # grad c_ij wrt c_i is (c_i - c_j)/dist_ij
    # Let's compute sum_j w_ij * (c_i - c_j)/dist_ij for each i
    
    # term_ij = w_ij * (c_i - c_j) / dist_ij
    # Sum over j for fixed i: sum_j w_ij * (c_i - c_j) / dist_ij
    
    # We can compute this as:
    # G_i = c_i * sum_j w_ij/dist_ij - sum_j w_ij * c_j / dist_ij
    
    # Let's construct a matrix M where M[i,j] = w_ij / dist_ij
    # Note: dist[i,i] is 0, but w[i,i] is 0, so 0/0 is NaN. Handle carefully.
    # Set dist diagonal to 1 to avoid division by zero, weight is 0 anyway.
    safe_dist = dist.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    M = w_mat / safe_dist
    
    # Sum_j M[i,j]
    sum_M = np.sum(M, axis=1)
    
    # Sum_j M[i,j] * c_j
    # M @ centers
    weighted_centers = M @ centers
    
    # grad_pair_part = c_i * sum_M[i] - weighted_centers[i]
    grad_pair_part = centers * sum_M[:, np.newaxis] - weighted_centers
    
    # The total gradient of the objective is - sum(w_i * grad_c_i)
    # So we add -grad_pair_part
    
    # 2. Boundary gradients
    w_bounds = weights[len(pair_dists):]
    
    # w_bounds indices:
    # 0..N-1: d_left (x+0.5) -> grad x = 1
    # N..2N-1: d_right (0.5-x) -> grad x = -1
    # 2N..3N-1: d_bottom (y+0.5) -> grad y = 1
    # 3N..4N-1: d_top (0.5-y) -> grad y = -1
    
    grad_bound_part = np.zeros_like(centers)
    
    # Left
    grad_bound_part[:, 0] += w_bounds[:N]
    # Right
    grad_bound_part[:, 0] -= w_bounds[N:2*N]
    # Bottom
    grad_bound_part[:, 1] += w_bounds[2*N:3*N]
    # Top
    grad_bound_part[:, 1] -= w_bounds[3*N:4*N]
    
    # Total gradient of objective (val)
    grad_val = - (grad_pair_part + grad_bound_part)
    
    return val, grad_val.flatten()

def _soft_min_optimize(centers, beta_start=50.0, beta_end=500.0, max_iter=100, deadline=0):
    """
    Optimize using L-BFGS-B on the soft-min objective.
    """
    if time.time() > deadline - 5.0:
        return centers
        
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # We can anneal beta. Start with lower beta (smoother) and increase (sharper).
    # L-BFGS-B doesn't support changing the function easily mid-run without restarting.
    # We can do multiple runs with increasing beta.
    
    best_x = x0
    best_val = soft_min_objective(x0, beta_start)[0]
    
    # Simple annealing schedule
    n_steps = 3
    betas = np.linspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                lambda x, b=beta: soft_min_objective(x, b)[0],
                best_x,
                jac=lambda x, b=beta: soft_min_objective(x, b)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            new_val = res.fun
            # Note: res.fun is the soft-min value. We want to minimize it (which maximizes the radius).
            # Wait, soft_min_objective returns val = lse/beta which is approx -min(c).
            # We want to maximize min(c), so we minimize -min(c).
            # So yes, we minimize val.
            
            if new_val < best_val:
                best_val = new_val
                best_x = res.x
                
        except Exception:
            pass
            
    return best_x.reshape(N, 2)

def _simulated_annealing(centers, start_temp=0.05, end_temp=0.001, steps=500, deadline=0):
    """
    Simulated Annealing to escape local minima.
    """
    current_centers = centers.copy()
    current_r = compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    T = start_temp
    
    for i in range(steps):
        if time.time() > deadline - 5.0:
            break
            
        # Generate a perturbation
        # Gaussian noise
        noise = np.random.normal(0, T, size=(N, 2))
        new_centers = current_centers + noise
        
        # Keep inside bounds? 
        # If we allow going outside, compute_radius returns small/negative.
        # It's better to clamp or reject.
        # Let's clamp to [-0.49, 0.49] to ensure validity for radius calc, 
        # but actually compute_radius handles outside by returning small r.
        # However, if r becomes negative, we treat it as a very bad move.
        
        new_r = compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Note: r is positive. Delta is negative.
            delta = new_r - current_r
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_centers = new_centers
                    current_r = new_r
        
        # Cool down
        T *= 0.995
        if T < end_temp:
            T = end_temp
            
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Initial Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    best_radius = compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Grid -> Soft-Min Optimize
    # 2. Simulated Annealing on the result
    # 3. Soft-Min Optimize again
    # 4. Repeat with different initializations
    
    def refine_pipeline(centers):
        # Step 1: Soft-min optimization
        c1 = _soft_min_optimize(centers, beta_start=50.0, beta_end=200.0, max_iter=50, deadline=deadline)
        
        # Step 2: Simulated Annealing
        c2 = _simulated_annealing(c1, start_temp=0.02, end_temp=0.001, steps=200, deadline=deadline)
        
        # Step 3: Final Soft-min optimization with higher beta
        c3 = _soft_min_optimize(c2, beta_start=200.0, beta_end=500.0, max_iter=100, deadline=deadline)
        
        return c3
    
    # Phase 1: Grid
    if time.time() < deadline - 30.0:
        c = refine_pipeline(initial_centers)
        r = compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
    
    # Phase 2: Randomized restarts
    max_restarts = 5
    for i in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Random initialization
        # Use a jittered grid or random points
        if i % 2 == 0:
            # Jittered grid
            c = initial_centers + rng.normal(0, 0.05, size=initial_centers.shape)
        else:
            # Random points, but ensure they are reasonably spread?
            # Just random uniform in [-0.4, 0.4]
            c = rng.uniform(-0.4, 0.4, size=(N, 2))
            
        c = refine_pipeline(c)
        r = compute_radius(c)
        if r > best_radius:
            best_radius = r
            best_centers = c
            
    return best_centers
