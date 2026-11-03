import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # We want to pack N=300 points.
    # Hexagonal packing density is ~0.9069.
    # Area of unit circle is pi. Area per circle ~ pi/300.
    # Side of hexagon cell s ~ sqrt(2 * pi / (300 * sqrt(3)))
    # Let's just generate a large grid and pick the best N.
    
    def generate_hex_lattice():
        # Estimate spacing
        # N ~ pi * R^2 / (s^2 * sqrt(3)/2) -> s ~ sqrt(2 * pi * R^2 / (N * sqrt(3)))
        # R=1, N=300
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        points = []
        # Generate grid
        max_k = int(1.0 / s) + 2
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
             # If not enough, shrink spacing
             while len(points) < N:
                 s *= 0.95
                 points = []
                 max_k = int(1.0 / s) + 2
                 for j in range(-max_k, max_k + 1):
                     for i in range(-max_k, max_k + 1):
                         x = (i + 0.5 * (j % 2)) * s
                         y = j * s * np.sqrt(3) / 2.0
                         if x*x + y*y <= 1.0:
                             points.append([x, y])
                 points = np.array(points)
        
        # Select N points closest to center to ensure compactness
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)
        return points[indices[:N]]

    P = generate_hex_lattice()
    
    # 2. Optimization
    # Objective: Minimize -L(P, beta)
    # L(P, beta) = -1/beta * log( sum_{i<j} exp(-beta * d_ij / 2) + sum_k exp(-beta * (1 - ||p_k||)) )
    
    def log_sum_exp_obj_and_grad(P_flat, beta):
        P = P_flat.reshape(-1, 2)
        
        # Pairwise distances
        # D[i, j] = ||P[i] - P[j]||
        # To save memory and time, we can compute gradients directly without full D matrix if N is small, 
        # but N=300 -> 45k pairs, which is fine.
        
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        
        # Mask upper triangle for pairs i < j
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        # Boundary distances
        norms = np.linalg.norm(P, axis=1)
        # Avoid division by zero in gradient
        norms_safe = np.where(norms > 1e-8, norms, 1e-8)
        boundary_dists = 1.0 - norms
        
        # Terms
        # We want to maximize the minimum, so we minimize the "soft min".
        # Soft min approx: -1/beta * log( sum exp(-beta * x) )
        # x_i are the distances (d_ij/2 and b_k)
        
        # Scale distances for the exponent
        # Note: The objective in the prompt is min( d_ij/2, b_k ).
        # So x_pairs = d_ij / 2
        # x_bound = b_k
        
        x_pairs = pair_dists / 2.0
        x_bound = boundary_dists
        
        # Combine all x's
        # To avoid overflow in exp(-beta * x), we can shift by min_x
        all_x = np.concatenate([x_pairs, x_bound])
        min_x = np.min(all_x)
        
        # Calculate sum of exp(-beta * (x - min_x))
        # This is numerically stable
        exponents = -beta * (all_x - min_x)
        # Clip exponents to prevent underflow/overflow issues if beta is huge, though stable shift helps
        exponents = np.clip(exponents, -700, 700)
        weights = np.exp(exponents)
        sum_weights = np.sum(weights)
        
        # Objective value: -L = 1/beta * log(sum_weights) - min_x
        # Wait, L = -1/beta * log( sum exp(-beta x) )
        # sum exp(-beta x) = exp(-beta min_x) * sum exp(-beta (x - min_x))
        # log(sum exp(-beta x)) = -beta min_x + log(sum exp(-beta (x - min_x)))
        # L = -1/beta * (-beta min_x + log(sum_w)) = min_x - 1/beta * log(sum_w)
        # We want to MAXIMIZE L. So we MINIMIZE -L.
        # -L = -min_x + 1/beta * log(sum_w)
        
        obj = -min_x + (1.0 / beta) * np.log(sum_weights)
        
        # Gradient
        # d/dP_m (-L) = - d/dP_m (L)
        # L = min_x - 1/beta * log(sum_w)
        # dL/dP_m = 0 (since min_x is just a scalar shift for stability, but actually min_x depends on P? 
        # No, we used min_x for numerical stability. The mathematical L is independent of the shift.
        # The shift is a constant for the derivative calculation if we treat min_x as fixed during the derivative step?
        # Actually, min_x is a function of P. But the log-sum-exp formula is smooth.
        # Let's use the standard gradient of the log-sum-exp soft-min.
        
        # Let S = sum_j w_j, where w_j = exp(-beta * x_j).
        # L = -1/beta * log(S)
        # dL/dx_j = -1/beta * (1/S) * dw_j/dx_j = -1/beta * (1/S) * w_j * (-beta) = w_j / S
        # So dL/dx_j = p_j, where p_j = w_j / S (probability distribution)
        
        # We want dL/dP_m.
        # dL/dP_m = sum_j (dL/dx_j) * (dx_j/dP_m)
        
        # For pairs (i, j): x_ij = ||P_i - P_j|| / 2
        # dx_ij/dP_i = (P_i - P_j) / (2 * ||P_i - P_j||) = (P_i - P_j) / (2 * d_ij)
        # dx_ij/dP_j = (P_j - P_i) / (2 * ||P_i - P_j||) = (P_j - P_i) / (2 * d_ij)
        
        # For boundary k: x_k = 1 - ||P_k||
        # dx_k/dP_k = - P_k / ||P_k||
        
        # Note: The indices in all_x correspond to:
        # 0..len(x_pairs)-1 : pairs
        # len(x_pairs)..N-1 : boundaries
        
        probs = weights / sum_weights
        
        grad_L = np.zeros_like(P)
        
        # Pair contributions
        # probs[0:len(x_pairs)] correspond to iu[0], iu[1]
        p_pairs = probs[:len(x_pairs)]
        
        # For each pair (i, j), contribution to grad_L[i] and grad_L[j]
        # dL/dP_i += p_ij * (P_i - P_j) / (2 * d_ij)
        # dL/dP_j += p_ij * (P_j - P_i) / (2 * d_ij)
        
        # Vectorized update
        # We need to sum over all pairs.
        # diff_vec = P[i] - P[j]
        # dist = d_ij
        # term = p_ij * diff_vec / (2 * dist)
        
        # Create arrays for i, j
        i_idx = iu[0]
        j_idx = iu[1]
        
        diff_vecs = P[i_idx] - P[j_idx]
        # dists are already computed in pair_dists
        # Avoid division by zero
        pair_dists_safe = np.where(pair_dists > 1e-8, pair_dists, 1e-8)
        
        # term shape: (num_pairs, 2)
        terms = p_pairs[:, np.newaxis] * diff_vecs / (2.0 * pair_dists_safe[:, np.newaxis])
        
        # Accumulate to grad_L
        # np.add.at is slow, but N=300, num_pairs=45k. It's okay.
        # Or use np.bincount style?
        # Let's use a simple loop or advanced indexing if possible.
        # Actually, we can use np.add.at for correctness.
        np.add.at(grad_L, i_idx, terms)
        np.add.at(grad_L, j_idx, -terms)
        
        # Boundary contributions
        p_bound = probs[len(x_pairs):]
        # x_k = 1 - ||P_k||
        # dx_k/dP_k = - P_k / ||P_k||
        # dL/dP_k += p_k * (- P_k / ||P_k||)
        
        bound_terms = -p_bound[:, np.newaxis] * P / norms_safe[:, np.newaxis]
        grad_L += bound_terms
        
        # We want to minimize -L, so gradient of objective is -grad_L
        grad_obj = -grad_L
        
        return obj, grad_obj

    # Annealing schedule
    betas = [20, 50, 100, 200, 400, 800]
    
    current_P = P
    best_P = P.copy()
    best_score = -np.inf
    
    def calculate_score(P_arr):
        P_arr = P_arr.reshape(-1, 2)
        diffs = P_arr[:, np.newaxis, :] - P_arr[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(N, 1)
        min_pair_dist = np.min(dists[iu])
        norms = np.linalg.norm(P_arr, axis=1)
        min_bound_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_bound_dist)

    for beta in betas:
        if time.time() - start_time > timeout:
            break
            
        P_flat = current_P.flatten()
        
        # Run optimization
        try:
            res = minimize(
                log_sum_exp_obj_and_grad, 
                P_flat, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 20}
            )
            new_P = res.x.reshape(-1, 2)
            
            # Validate
            if not np.all(np.isfinite(new_P)):
                continue
                
            # Check if it's better
            score = calculate_score(new_P)
            if score > best_score:
                best_score = score
                best_P = new_P.copy()
                current_P = new_P.copy()
                
        except Exception as e:
            # If optimization fails, keep current
            pass

    # Final check
    final_score = calculate_score(best_P)
    current_score = calculate_score(current_P)
    
    if final_score < current_score:
        best_P = current_P
    
    return best_P

# EVOLVE-BLOCK-END
