"""
Strategy: Custom Vectorized Gradient Descent with LSE Annealing.
Instead of scipy L-BFGS-B, we use a manual gradient descent loop with backtracking.
This allows for:
1. Faster iterations (no overhead of scipy's internal machinery).
2. Precise control over the LSE temperature (beta) annealing.
3. A final "polish" phase with very high beta to tighten the minimum distance.
"""

import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Compute the actual common radius r."""
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(0.5 - centers, centers + 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall_dist) / 2.0

def _lse_objective_and_grad(centers, beta):
    """
    Compute the LSE surrogate objective and its gradient.
    Objective: (1/beta) * log( sum_{i<j} exp(-beta * d_ij) + sum_i sum_k exp(-beta * d_ik_wall) )
    Note: The problem asks for r = min(d_ij/2, d_wall).
    So we want to maximize min(d_ij/2, d_wall).
    Let f = -r. We minimize f.
    The terms in the max are: -d_ij/2 and -d_wall.
    LSE approximates max.
    So we minimize (1/beta) log( sum exp(beta * (-d_ij/2)) + sum exp(beta * (-d_wall)) ).
    
    Let's stick to the previous formulation which was:
    Minimize F = (1/beta) log( sum exp(-beta * d_ij/2) + sum exp(-beta * d_wall_i) )
    Wait, d_wall_i is the min distance to any wall.
    So the set of distances D is { d_ij/2 for i<j } U { d_wall_i for i }.
    We want to maximize min(D).
    Equivalent to minimizing -min(D) = max(-D).
    LSE approximates max.
    So minimize (1/beta) log( sum_{k in D} exp(-beta * d_k) ).
    
    Here d_k are:
    1. d_ij / 2
    2. d_wall_i (which is min of 6 distances to faces)
    
    Gradient of F w.r.t centers:
    F = (1/beta) log(S), S = sum exp(-beta d_k)
    dF/dc = (1/beta) (1/S) dS/dc = (1/beta) (1/S) sum exp(-beta d_k) (-beta) (dd_k/dc)
          = - (1/S) sum exp(-beta d_k) (dd_k/dc)
    
    Let w_k = exp(-beta d_k) / S.
    grad = - sum w_k (dd_k/dc)
    
    For pair (i,j): d_k = ||c_i - c_j|| / 2.
    dd_k/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    dd_k/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    For wall i, face k: d_k = dist to face.
    dd_k/dc_i = unit vector pointing towards the face (i.e., away from center towards wall).
    If face is x=0.5, d = 0.5 - x, dd/dx = -1. Vector is [-1, 0, 0].
    If face is x=-0.5, d = x + 0.5, dd/dx = 1. Vector is [1, 0, 0].
    
    So grad = - sum w_k * (vector pointing from center to constraint boundary)
    This means the force pushes the center AWAY from the constraint boundary, which is correct for maximization.
    """
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu] / 2.0  # d_k = d_ij / 2
    pair_diffs = diff[iu]
    
    # t_k = -beta * d_k
    # To avoid underflow/overflow in exp, we shift by max t_k.
    # But we need weights w_k = exp(t_k) / S.
    # Let's compute log_weights = t_k - log_S.
    
    # Wall terms
    # d_wall_i = min(0.5 - x, x + 0.5, ...)
    # We need to treat each face separately in the LSE sum?
    # Yes, because max(-d_wall_i) = max( -(min d_faces) ) = max( -d_face1, -d_face2, ... ).
    # So the set D actually contains the individual face distances, not the min.
    # D = { d_ij/2 } U { d_face_ik for all i, k }.
    
    # Let's re-evaluate the terms.
    # Terms in S:
    # 1. For each pair i<j: exp(-beta * d_ij/2)
    # 2. For each sphere i and face k: exp(-beta * d_face_ik)
    
    # Calculate all d_k values
    d_pairs = pair_dists
    # d_faces: shape (N, 6)
    # d_x+ = 0.5 - x
    # d_x- = x + 0.5
    d_faces = np.zeros((n, 6))
    d_faces[:, 0] = 0.5 - centers[:, 0]
    d_faces[:, 1] = centers[:, 0] + 0.5
    d_faces[:, 2] = 0.5 - centers[:, 1]
    d_faces[:, 3] = centers[:, 1] + 0.5
    d_faces[:, 4] = 0.5 - centers[:, 2]
    d_faces[:, 5] = centers[:, 2] + 0.5
    
    d_all = np.concatenate([d_pairs, d_faces.flatten()])
    
    # Calculate log weights
    # t_k = -beta * d_k
    t_all = -beta * d_all
    
    # Stable log-sum-exp
    max_t = np.max(t_all)
    # If max_t is very large, exp might overflow. But t_k are negative (d_k > 0).
    # So max_t is close to 0 (if d is small) or very negative.
    # exp(t_k - max_t) is safe.
    
    log_sum_exp = max_t + np.log(np.sum(np.exp(t_all - max_t)))
    
    # Weights w_k = exp(t_k) / exp(log_sum_exp) = exp(t_k - log_sum_exp)
    w_all = np.exp(t_all - log_sum_exp)
    
    # Objective value
    obj = (1.0 / beta) * log_sum_exp
    
    # Gradient
    # grad = - sum w_k * (dd_k/dc)
    
    # Pairwise part
    w_pairs = w_all[:len(d_pairs)]
    # dd_ij/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # Note: pair_diffs = c_i - c_j.
    # unit_vecs = pair_diffs / ||c_i - c_j||
    # But we have d_pairs = ||c_i - c_j|| / 2. So ||c_i - c_j|| = 2 * d_pairs.
    # unit_vecs = pair_diffs / (2 * d_pairs)
    
    # Handle zero distances (should not happen if we jitter, but safe)
    safe_d_pairs = np.where(d_pairs < 1e-10, 1e-10, d_pairs)
    unit_vecs = pair_diffs / (2.0 * safe_d_pairs[:, np.newaxis])
    
    # contrib to grad[i] from pair (i,j): -w * 0.5 * (c_i - c_j)/d_ij = -w * 0.5 * unit_vecs * 2 = -w * unit_vecs?
    # Let's check derivative of d_ij/2.
    # d/dc_i (||c_i - c_j||/2) = 0.5 * (c_i - c_j)/||c_i - c_j||
    # So dd_k/dc_i = 0.5 * unit_vecs.
    # grad += -w * (0.5 * unit_vecs)
    
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, 0.5 * w_pairs[:, np.newaxis] * unit_vecs)
    
    # Wall part
    w_faces = w_all[len(d_pairs):].reshape(n, 6)
    
    # Gradients for faces:
    # d_x+ = 0.5 - x -> dd/dx = -1
    # d_x- = x + 0.5 -> dd/dx = 1
    # d_y+ = 0.5 - y -> dd/dy = -1
    # d_y- = y + 0.5 -> dd/dy = 1
    # d_z+ = 0.5 - z -> dd/dz = -1
    # d_z- = z + 0.5 -> dd/dz = 1
    
    # grad_x += -w_x+ * (-1) + -w_x- * (1) = w_x+ - w_x-
    # grad_y += w_y+ - w_y-
    # grad_z += w_z+ - w_z-
    
    grad[:, 0] += w_faces[:, 0] - w_faces[:, 1]
    grad[:, 1] += w_faces[:, 2] - w_faces[:, 3]
    grad[:, 2] += w_faces[:, 4] - w_faces[:, 5]
    
    return obj, grad

def _gradient_descent_step(centers, beta, max_iter=1000, tol=1e-12, time_limit=None, start_time=None):
    """
    Perform gradient descent with backtracking line search.
    """
    if start_time is None:
        start_time = time.time()
    
    x = centers.copy()
    best_obj, best_grad = _lse_objective_and_grad(x, beta)
    best_x = x.copy()
    
    # Adaptive learning rate
    lr = 0.01
    min_lr = 1e-8
    
    for it in range(max_iter):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        obj, grad = _lse_objective_and_grad(x, beta)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        # Gradient descent step
        # We want to minimize obj. So step is -grad.
        # Normalize gradient to stabilize step size?
        # Let's try simple step first.
        
        # Backtracking line search
        alpha = lr
        obj_new = obj
        x_new = x
        improved = False
        
        for k in range(20):
            x_candidate = x - alpha * grad
            # Clip to bounds
            x_candidate = np.clip(x_candidate, -0.5, 0.5)
            
            obj_candidate, _ = _lse_objective_and_grad(x_candidate, beta)
            
            # Armijo condition
            if obj_candidate < obj_new - 1e-4 * alpha * np.dot(grad, x - x_candidate):
                # Actually, standard Armijo: f(x + alpha * d) <= f(x) + c * alpha * grad^T d
                # Here d = -grad. So grad^T d = -||grad||^2.
                # Condition: f_new <= f_old - c * alpha * ||grad||^2
                if obj_candidate < obj - 1e-4 * alpha * np.dot(grad, grad):
                    x = x_candidate
                    obj = obj_candidate
                    improved = True
                    break
            alpha *= 0.5
            
        if not improved:
            # If no improvement, reduce learning rate
            lr *= 0.5
            if lr < min_lr:
                break
        else:
            # If improved, we can keep lr or increase slightly if we had to backtrack a lot?
            # Keep simple: keep lr.
            pass
        
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
            
    return best_x

def _generate_initial_grid():
    """Generate a 4x4x4 grid, take first 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_fcc_like():
    """
    Generate a more structured start.
    Use a 3x4x5 grid? 60 points.
    Or just jittered grid.
    """
    base = _generate_initial_grid()
    # Add small random jitter to break symmetry
    rng = np.random.RandomState(42)
    base = base + rng.randn(N, 3) * 0.01
    return np.clip(base, -0.5, 0.5)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_centers = None
    best_radius = -np.inf
    
    # Multi-start
    initializations = [
        _generate_initial_grid,
        _generate_fcc_like
    ]
    
    # Annealing schedule
    # Start with low beta (smooth), go to high beta (sharp)
    # Betas: 10, 50, 100, 200, 500
    
    current_centers = None
    
    for init_func in initializations:
        if time.time() > deadline - 20:
            break
            
        init_centers = init_func()
        
        # We will run the annealing process from this start
        c = init_centers.copy()
        
        # Annealing steps
        # Allocate time for each beta stage
        # Total time available ~ 145s.
        # 2 starts * 5 betas = 10 stages.
        # ~10s per stage?
        
        betas = [10.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            time_left = deadline - time.time()
            if time_left < 5:
                break
                
            # Allocate time for this beta stage
            # Later betas need more precision, so maybe more time?
            # But they converge faster locally?
            # Let's give equal time, or slightly more for high beta.
            t_alloc = min(15.0, time_left / len(betas))
            
            c = _gradient_descent_step(c, beta, max_iter=500, time_limit=t_alloc, start_time=start_time)
            
            # Check radius
            r = _compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_centers = c.copy()
                
        # If we improved significantly, use this as the new current best for potential further refinement?
        # For now, just keep the global best.
        
    # Final Polish
    # If we have time, run a very high beta optimization on the best centers found
    if best_centers is not None and time.time() < deadline - 5:
        # Use a very high beta to polish
        beta_polish = 1000.0
        t_polish = deadline - time.time() - 1
        if t_polish > 1:
            polished = _gradient_descent_step(best_centers, beta_polish, max_iter=200, time_limit=t_polish, start_time=start_time)
            r_polish = _compute_radius(polished)
            if r_polish > best_radius:
                best_radius = r_polish
                best_centers = polished
    
    if best_centers is None:
        best_centers = _generate_initial_grid()
        
    # Final safety clip
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# Ensure the function is callable
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Radius: {_compute_radius(res)}")
