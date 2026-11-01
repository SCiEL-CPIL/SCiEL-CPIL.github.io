"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _soft_min(values, tau):
    """
    Compute a soft minimum of values.
    soft_min = -1/tau * log(sum(exp(-tau * values)))
    """
    # Shift values to avoid overflow in exp
    v_min = np.min(values)
    shifted = values - v_min
    # If tau is very large, this approaches min
    # If tau is small, this approaches mean
    log_sum_exp = np.log(np.sum(np.exp(-tau * shifted)))
    return v_min - (1.0 / tau) * log_sum_exp

def _ring_initialization(n):
    """
    Initialize points in concentric rings.
    """
    points = []
    
    # Heuristic for ring counts.
    # For hexagonal packing, ring k has 6k points.
    # Total points = 1 + sum(6k) = 1 + 3k(k+1).
    # For N=300, 3k(k+1) ~ 299 => k(k+1) ~ 100 => k ~ 10.
    # Rings: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54 ...
    # Sum up to 10: 1 + 6*55 = 331. Too many.
    # Sum up to 9: 1 + 6*45 = 271. Need 29 more.
    # So rings: 1, 6, 12, 18, 24, 30, 36, 42, 48, 45 (last ring partial).
    
    ring_counts = [1]
    remaining = n - 1
    k = 1
    while remaining > 0:
        count = min(6 * k, remaining)
        ring_counts.append(count)
        remaining -= count
        k += 1
        
    # Estimate radii.
    # Ideal radius r ~ 0.053.
    # Ring radius R_k should be roughly k * 2 * r * sin(60 deg) ? 
    # In hex packing, radial distance between rings is sqrt(3) * r.
    # Let's start with an estimate and let the optimizer fix it.
    # r_est = 0.053
    r_est = 0.053
    radial_step = np.sqrt(3) * r_est
    
    for i, count in enumerate(ring_counts):
        if count == 0:
            continue
        if i == 0:
            points.append([0.0, 0.0])
        else:
            radius = i * radial_step
            # Add a small random jitter to break symmetry if needed, 
            # but for ring initialization, regular spacing is good.
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Offset alternate rings by pi/count to interlock?
            # Hex packing: ring i is offset by (i % 2) * pi / count ?
            # Actually, just regular spacing is a good start.
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            points.extend(zip(xs, ys))
            
    points = np.array(points)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization
    points = _ring_initialization(N)
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # 2. Optimization using L-BFGS-B with soft-min objective
    # We anneal the temperature tau from low (smooth, global-ish) to high (sharp, local).
    
    def objective_and_grad(x, tau):
        p = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        # We need distances, not sq distances, for the min radius.
        # But for soft min, we can work with distances.
        # To save time, we only compute sqrt for non-inf values?
        # N=300, 300x300 matrix is 90k elements. Sqrt is cheap.
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        boundary_dists = 1.0 - norms
        
        # Collect all constraints:
        # Pairwise: dists[i,j] / 2
        # Boundary: boundary_dists[i]
        
        # To avoid constructing a huge vector of 90k + 300 elements every time,
        # we can compute the soft min directly.
        
        # Soft min of pairwise distances / 2
        # Let d_ij = dists[i,j]. We want min(d_ij/2).
        # soft_min(d/2) = -1/tau * log(sum(exp(-tau * d/2)))
        
        # We can compute the log-sum-exp efficiently.
        # Note: dists has inf on diagonal. exp(-inf) = 0. So it's safe.
        
        # Pairwise term
        # We sum over i < j to avoid double counting? 
        # The min is the same. The sum in log-sum-exp is over all elements.
        # If we sum over all i,j (i!=j), we double count the terms.
        # min(A, A, B) is same as min(A, B).
        # soft_min(A, A, B) is NOT the same as soft_min(A, B) if we care about the exact value, 
        # but for the purpose of finding the location of the minimum, the gradient direction is similar.
        # However, the magnitude of the "effective" constraint might be skewed.
        # Let's stick to i < j for correctness of the soft min value relative to the hard min.
        
        # Extract upper triangle
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary
        bound_dists = boundary_dists
        
        # Combine
        # To compute soft min of two sets, we can combine the log-sum-exps.
        # soft_min(A, B) = -1/tau * log( sum(exp(-tau A)) + sum(exp(-tau B)) )
        
        # Calculate log-sum-exp for pairs
        # Shift to prevent underflow/overflow
        min_pair = np.min(pair_dists)
        min_bound = np.min(bound_dists)
        
        # We want to compute:
        # LSE = log( sum(exp(-tau * (pair_dists - min_pair))) * exp(-tau * min_pair) + sum(exp(-tau * (bound_dists - min_bound))) * exp(-tau * min_bound) )
        # This is getting complex. 
        # Simpler: Just create the vector. 45k + 300 floats is ~ 180KB. 
        # Creating it 1000 times is 180MB. This is fine for 150s.
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        # Soft min
        min_val = np.min(all_dists)
        shifted = all_dists - min_val
        log_sum_exp = np.log(np.sum(np.exp(-tau * shifted)))
        soft_min_val = min_val - (1.0 / tau) * log_sum_exp
        
        # Gradient
        # d/dx soft_min = - (1/tau) * ( sum( exp(-tau x_i) * d(-tau x_i)/dx ) / sum( exp(-tau x_i) ) )
        # = sum( w_i * grad(x_i) ) where w_i = exp(-tau x_i) / sum(exp(-tau x_j))
        # Note: x_i are the distances.
        
        # We need the weights w for each constraint.
        # w_i = exp(-tau * (x_i - min_val)) / sum(exp(-tau * (x_j - min_val)))
        # This is a softmax over -x.
        
        exp_terms = np.exp(-tau * shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Now, we need to propagate these weights back to the points.
        # The constraints are:
        # 1. Pairwise: d_ij/2. Index in all_dists: 0 to len(pair_dists)-1.
        # 2. Boundary: b_i. Index in all_dists: len(pair_dists) to end.
        
        grad = np.zeros_like(p)
        
        # Pairwise contributions
        # d_ij = |p_i - p_j|. 
        # d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij)
        # d/dp_j (d_ij/2) = (p_j - p_i) / (2 * d_ij)
        
        # We have weights for each pair (i,j) in the upper triangle.
        # We need to map back to (i,j) indices.
        # iu[0] is i, iu[1] is j.
        
        # Vectorized update:
        # For each pair k with weight w_k, indices i_k, j_k:
        # grad[i_k] += w_k * (p[i_k] - p[j_k]) / (2 * d_k)
        # grad[j_k] -= w_k * (p[i_k] - p[j_k]) / (2 * d_k)
        
        # Note: d_k here is the distance, not dist/2.
        # The constraint value is d_k/2.
        # The gradient of the constraint value is (p_i - p_j) / (2 * d_k).
        
        # Let's compute the unit vectors and scale by weights.
        # diff = p_i - p_j
        # unit = diff / d_k
        # grad_contrib = w_k * unit / 2
        
        # We can use np.add.at for accumulation.
        
        # Get indices
        i_idx, j_idx = iu
        w_pairs = weights[:len(pair_dists)]
        
        # Distances for normalization
        # d_k = 2 * pair_dists[k]
        d_pairs = 2.0 * pair_dists
        
        # Avoid division by zero
        mask = d_pairs > 1e-12
        w_pairs_safe = np.where(mask, w_pairs, 0.0)
        d_pairs_safe = np.where(mask, d_pairs, 1.0)
        
        # Unit vectors
        # diff = p[i] - p[j]
        diff_vec = p[i_idx] - p[j_idx]
        unit_vecs = diff_vec / d_pairs_safe[:, np.newaxis]
        
        # Contribution magnitude
        # w_k / (2 * d_k) ? No.
        # Grad of (d/2) is (p_i - p_j) / (2 d).
        # So vector is unit_vec * (w_k / 2) ?
        # Wait. d/dp_i (d_ij/2) = (p_i - p_j) / (2 * d_ij).
        # The vector (p_i - p_j) / d_ij is the unit vector u.
        # So gradient is u / 2.
        # Weighted gradient: w_k * u_k / 2.
        
        coeffs = w_pairs_safe / 2.0
        
        # Accumulate to grad
        # grad[i] += coeff * u
        # grad[j] -= coeff * u
        
        np.add.at(grad, i_idx, coeffs[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -coeffs[:, np.newaxis] * unit_vecs)
        
        # Boundary contributions
        # Constraint: 1 - |p_i|.
        # Grad: -p_i / |p_i|.
        
        w_bounds = weights[len(pair_dists):]
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        bound_grads = -p / norms_safe[:, np.newaxis]
        
        # Weighted sum
        grad += w_bounds[:, np.newaxis] * bound_grads
        
        # The objective is -soft_min.
        # So grad_obj = -grad_soft_min.
        # Wait, the formula for grad soft_min gave us the gradient of the soft_min value.
        # We want to minimize -soft_min.
        # So we return -soft_min_val, -grad.
        
        return -soft_min_val, -grad

    # Annealing loop
    # Start with low tau (smooth), end with high tau (sharp)
    taus = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    current_points = points.flatten()
    
    for tau in taus:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(
                lambda x: objective_and_grad(x, tau)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, tau)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-6,
                    'ftol': 1e-10,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    current_points = candidate.flatten()
                else:
                    # If it didn't improve the hard score, keep the previous best for next iteration
                    # or use the candidate if it's closer to the target?
                    # Usually, if hard score didn't improve, the soft min optimization might have moved
                    # away from the optimal hard min. 
                    # However, if we are annealing, we want to follow the soft min.
                    # Let's use the candidate as the start for the next (sharper) tau,
                    # but keep best_points as the one with highest hard score.
                    current_points = candidate.flatten()
                    
        except Exception as e:
            # If optimization fails, break
            break
            
    # Final polish with a very high tau (effectively hard min)
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda x: objective_and_grad(x, 1000.0)[0],
                current_points,
                jac=lambda x: objective_and_grad(x, 1000.0)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-8,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            if time.time() < deadline:
                candidate = res_final.x.reshape(-1, 2)
                candidate_score = _compute_score(candidate)
                if candidate_score > best_score:
                    best_points = candidate
                    best_score = candidate_score
                    
        except Exception as e:
            pass

    # Final check
    norms = np.linalg.norm(best_points, axis=1)
    # Ensure points are inside
    # If any point is outside, scale it back? 
    # The objective penalizes going outside, so it should be fine.
    
    return best_points

# EVOLVE-BLOCK-END
