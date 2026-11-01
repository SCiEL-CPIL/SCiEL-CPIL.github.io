import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # 1. Initialization
    # Ring counts: 1, 6, 12, 18, 13
    ring_counts = [1, 6, 12, 18, 13]
    centers = []
    
    # Estimate optimal radius roughly for init
    r_est = 0.125
    
    # Radii for rings
    # Outer ring should be close to 1 - r_est
    R_max = 1.0 - r_est
    # Distribute radii roughly linearly
    radii = [i * R_max / (len(ring_counts) - 1) for i in range(len(ring_counts))]
    # Fix center at 0
    radii[0] = 0.0
    
    for k, count in enumerate(ring_counts):
        R = radii[k]
        if count == 1:
            centers.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                centers.append([R * np.cos(a), R * np.sin(a)])
                
    centers = np.array(centers)
    
    # Add small perturbation to break symmetry
    np.random.seed(42) # Reproducibility
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Clip to inside container to ensure validity at start
    dists = np.linalg.norm(centers, axis=1)
    mask = dists > (1.0 - 0.1) # Keep them well inside
    centers[mask] /= dists[mask, np.newaxis] * (1.0 - 0.1)
    
    best_score = 0.0
    best_centers = centers.copy()
    
    # 2. Optimization
    beta = 10.0
    
    def soft_min_obj_and_grad(x):
        # x is flattened centers (2N)
        c = x.reshape(-1, 2)
        
        # Compute all v_k
        # Pairwise distances
        # c_i - c_j
        # Using broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Only upper triangle for pairs
        i, j = np.triu_indices(N, k=1)
        pair_dists = dists[i, j]
        v_pairs = pair_dists / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(c, axis=1)
        v_bounds = 1.0 - norms
        
        # Combine
        v_all = np.concatenate([v_pairs, v_bounds])
        
        # Soft min
        # g = 1/beta * ln(sum(exp(-beta * v)))
        # Numerical stability: subtract max v? No, we want soft min, so large v contributes less.
        # exp(-beta * v). If v is large, exp is small.
        # To avoid underflow, we can just compute it. v is roughly 0.1 to 1.0.
        # beta is 10 to 1000. beta * v is 1 to 1000.
        # exp(-1000) is 0. This is fine.
        
        exp_terms = np.exp(-beta * v_all)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            obj = 0.0
            grad = np.zeros_like(x)
            return obj, grad
            
        weights = exp_terms / sum_exp
        obj = (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # grad_g = - sum_k w_k * grad_v_k
        
        grad_c = np.zeros_like(c)
        
        # Boundary gradients
        # v_i = 1 - ||c_i||
        # grad_v_i = - c_i / ||c_i||
        # w_bound_i is the weight for v_bounds[i]
        w_bounds = weights[-N:]
        
        # Avoid division by zero
        safe_norms = np.where(norms < 1e-8, 1e-8, norms)
        grad_bounds = -c / safe_norms[:, np.newaxis]
        grad_c += w_bounds[:, np.newaxis] * grad_bounds
        
        # Pair gradients
        # v_ij = 0.5 * ||c_i - c_j||
        # grad_{c_i} v_ij = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # grad_{c_j} v_ij = -0.5 * (c_i - c_j) / ||c_i - c_j||
        
        w_pairs = weights[:len(v_pairs)]
        
        # We need to accumulate gradients for each pair
        # diff[i, j] = c_i - c_j
        # dists[i, j] = ||c_i - c_j||
        
        # Iterate or vectorize?
        # Vectorizing accumulation is tricky.
        # Let's use a loop for the pairs since N=50 is small (1225 pairs).
        # Or better, use the fact that grad_c_i += sum_j w_ij * 0.5 * (c_i - c_j)/d_ij
        
        # Let's compute the vector contribution
        # For each pair (i, j):
        # vec = 0.5 * (c_i - c_j) / d_ij
        # grad_c[i] += w_ij * vec
        # grad_c[j] -= w_ij * vec
        
        # We can do this with advanced indexing and np.add.at
        # But simple loop is probably fast enough and less error prone for 1225 items.
        
        for idx in range(len(v_pairs)):
            i_idx = i[idx]
            j_idx = j[idx]
            d = pair_dists[idx]
            if d < 1e-8: continue
            vec = 0.5 * (c[i_idx] - c[j_idx]) / d
            w = w_pairs[idx]
            grad_c[i_idx] += w * vec
            grad_c[j_idx] -= w * vec
            
        grad = grad_c.flatten()
        
        return obj, grad

    # Optimization loop
    current_centers = centers.copy()
    
    while time.time() < deadline:
        res = minimize(
            soft_min_obj_and_grad, 
            current_centers.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.status == 1:
            current_centers = res.x.reshape(-1, 2)
            
            # Calculate actual score
            c = current_centers
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            i, j = np.triu_indices(N, k=1)
            min_pair_dist = np.min(dists[i, j])
            norms = np.linalg.norm(c, axis=1)
            min_bound_dist = np.min(1.0 - norms)
            
            r = min(min_pair_dist / 2.0, min_bound_dist)
            
            if r > best_score:
                best_score = r
                best_centers = current_centers.copy()
                
            # Increase beta
            beta *= 2.0
            if beta > 1000:
                break
        else:
            # If failed, try again with slightly different start? Or break?
            # Just break if we can't improve
            break

    # Final check
    c = best_centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dists[i, j])
    norms = np.linalg.norm(c, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    # Ensure validity
    if min_pair_dist < 1e-6 or min_bound_dist < -1e-6:
        # Fallback to seed logic if something went wrong
        return _fallback()
        
    return best_centers

def _fallback():
    # Hex lattice from seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98
