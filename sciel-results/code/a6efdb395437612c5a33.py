"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _soft_min_objective_and_grad(C, beta):
    """
    Compute the soft-min objective and its gradient with respect to C.
    C: (N, 2) array of centers.
    beta: sharpness parameter.
    Returns: objective_value, gradient (N, 2)
    """
    N = C.shape[0]
    # Compute all pairwise distances
    # dist[i, j] = |C[i] - C[j]|
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal to avoid division by zero later if needed, 
    # but we will construct specific gap vectors.
    
    # Gaps:
    # 1. Pairwise distances (i < j)
    # 2. Boundary distances: 1 - |C[i]|
    
    # We need to handle the set of all gaps.
    # To be efficient, we can compute the weights for pairs and boundaries separately 
    # and combine them, but the log-sum-exp normalization is global.
    
    # Let's construct the vector of all gaps x_k.
    # Number of pairs: N*(N-1)/2
    # Number of boundaries: N
    
    # For N=50, pairs = 1225, boundaries = 50. Total 1275.
    # This is small enough to keep in memory.
    
    # Extract upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dist[i_idx, j_idx]
    
    # Boundary gaps
    radii = np.linalg.norm(C, axis=1)
    bound_gaps = 1.0 - radii
    
    # Combine gaps
    # Note: The score is min(min_pair_dist/2, min_bound_gap).
    # We want to maximize this.
    # Let's define the objective as the soft-min of:
    # { pair_dists / 2 } and { bound_gaps }
    # This ensures the scale is consistent with the final radius r.
    
    scaled_pair_dists = pair_dists / 2.0
    
    all_gaps = np.concatenate([scaled_pair_dists, bound_gaps])
    
    # Compute weights p_k = exp(-beta * x_k) / sum(exp(-beta * x_m))
    # To avoid underflow/overflow, subtract max
    max_gap = np.max(all_gaps)
    exp_terms = np.exp(-beta * (all_gaps - max_gap))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value
    obj = max_gap - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    grad = np.zeros_like(C)
    
    # Pairwise contributions
    # For each pair (i, j), weight w_k
    # d/dC_i (|C_i - C_j|/2) = (C_i - C_j) / (2 * |C_i - C_j|)
    # d/dC_j (|C_i - C_j|/2) = (C_j - C_i) / (2 * |C_i - C_j|)
    
    # Vectorized update for pairs
    # diff_pair = C[i_idx] - C[j_idx]
    # norm_pair = pair_dists
    # unit_vec = diff_pair / norm_pair[:, np.newaxis]
    # grad[i] += weights * unit_vec
    # grad[j] -= weights * unit_vec
    
    # We use np.add.at for accumulation
    pair_weights = weights[:len(pair_dists)]
    diff_pair = C[i_idx] - C[j_idx]
    # Handle zero distance (shouldn't happen with random init, but safe)
    mask = pair_dists > 1e-10
    unit_vec = np.zeros_like(diff_pair)
    unit_vec[mask] = diff_pair[mask] / pair_dists[mask, np.newaxis]
    
    # Scale by 1/2 because gap was dist/2
    # Gradient of dist/2 is 0.5 * unit_vec
    contrib = 0.5 * pair_weights[:, np.newaxis] * unit_vec
    
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    # Boundary contributions
    # Gap = 1 - |C_i|
    # d/dC_i (1 - |C_i|) = -C_i / |C_i|
    
    bound_weights = weights[len(pair_dists):]
    # Handle zero radius
    mask_b = radii > 1e-10
    unit_rad = np.zeros_like(C)
    unit_rad[mask_b] = C[mask_b] / radii[mask_b, np.newaxis]
    
    contrib_b = -bound_weights[:, np.newaxis] * unit_rad
    grad += contrib_b
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_score = 0.0
    best_C = None
    
    # Parameters for optimization
    beta_start = 10.0
    beta_end = 500.0
    num_restarts = 20
    
    # Precompute indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for restart in range(num_restarts):
        # Initialize C
        # Use a rough hexagonal lattice or random points?
        # Random points inside a smaller circle to avoid immediate boundary issues
        # Or a spiral?
        # Let's try a jittered grid or random.
        # Random uniform in unit disk
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.8 # Keep away from boundary initially
        C = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Alternatively, a hex lattice is a good start
        if restart == 0:
             # Seed-like init
             spacing = 0.2
             points = []
             for j in range(-5, 6):
                 for i in range(-5, 6):
                     x = (i + 0.5 * (j % 2)) * spacing
                     y = j * spacing * np.sqrt(3) / 2
                     if x*x + y*y < 0.8:
                         points.append([x, y])
             if len(points) >= N:
                 C = np.array(points[:N])
        
        beta = beta_start
        step_size = 0.01
        last_improve_iter = 0
        current_score = 0.0
        
        # Optimization loop
        max_iters = 2000
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute objective and gradient
            obj, grad = _soft_min_objective_and_grad(C, beta)
            
            # The true score is the min of the gaps.
            # We can estimate it from obj as beta -> inf.
            # For tracking progress, use a moderate beta or just the raw min.
            # Raw min is cheap for N=50.
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            pair_dists = dist[i_idx, j_idx]
            radii = np.linalg.norm(C, axis=1)
            bound_gaps = 1.0 - radii
            
            raw_score = np.min(np.concatenate([pair_dists / 2.0, bound_gaps]))
            
            if raw_score > current_score + 1e-8:
                current_score = raw_score
                last_improve_iter = it
                if raw_score > best_score:
                    best_score = raw_score
                    best_C = C.copy()
            
            # Update C
            C += step_size * grad
            
            # Adaptive step size / Annealing
            # Increase beta slowly to sharpen
            if it % 50 == 0:
                beta *= 1.1
                if beta > beta_end:
                    beta = beta_end
                step_size *= 0.95
                
            # Stagnation check
            if it - last_improve_iter > 200:
                # Perturb
                C += np.random.normal(0, 0.05, C.shape)
                last_improve_iter = it
                step_size = 0.01
                beta = max(beta_start, beta * 0.5)
                
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    # Final polish on best_C
    if best_C is not None:
        # Run a few more iterations with high beta on the best solution
        C = best_C.copy()
        beta = 1000.0
        step_size = 0.001
        for _ in range(500):
            if time.time() > deadline:
                break
            obj, grad = _soft_min_objective_and_grad(C, beta)
            C += step_size * grad
            # Check improvement
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            pair_dists = dist[i_idx, j_idx]
            radii = np.linalg.norm(C, axis=1)
            bound_gaps = 1.0 - radii
            raw_score = np.min(np.concatenate([pair_dists / 2.0, bound_gaps]))
            if raw_score > best_score:
                best_score = raw_score
                best_C = C.copy()
    
    return best_C
# EVOLVE-BLOCK-END
