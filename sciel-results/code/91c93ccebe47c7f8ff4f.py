import numpy as np
import time
import math
from scipy.spatial import KDTree
from scipy.optimize import minimize

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Use KDTree for fast nearest neighbor search to get min pair distance
    tree = KDTree(points)
    # Query 1 nearest neighbor for each point
    # The distance to the nearest neighbor is the min distance for that point
    # The global min pair distance is the min of these nearest neighbor distances
    # Note: KDTree query with k=1 returns the point itself, so k=2 is needed to get the *other* point
    # However, for score calculation, we just need the min distance.
    # Actually, for N=300, full pairwise is ~45k ops. KDTree is O(N log N).
    # Let's stick to KDTree for speed.
    try:
        # k=2 to avoid self-match
        dists, idxs = tree.query(points, k=2)
        # dists[:, 1] contains the distance to the nearest neighbor
        min_pair = np.min(dists[:, 1])
    except Exception:
        # Fallback to full pairwise if KDTree fails for some reason (e.g., duplicate points causing issues in some versions, though usually handled)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_full = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_full, np.inf)
        min_pair = np.min(dists_full)
        
    score = min(min_pair, min_bound)
    return score

def soft_min_value_and_grad(points, k_smooth=50.0):
    """
    Compute the soft-min of pairwise and boundary distances, and its gradient.
    This is used for the optimization step.
    """
    n = len(points)
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    # Get nearest neighbors to approximate the min
    tree = KDTree(points)
    # We need the actual distances to a few nearest neighbors to form the soft min
    # Using all pairs is too slow for gradient. 
    # Strategy: Use k-nearest neighbors (k=10) for each point to build a sparse interaction graph.
    # This is an approximation of the global min, but valid for local optimization.
    
    k_nn = 10
    try:
        dists, idxs = tree.query(points, k=k_nn+1) # +1 for self
        # dists shape (n, k_nn+1), idxs shape (n, k_nn+1)
    except Exception:
        # Fallback: just use boundary and a rough estimate
        min_d = np.min(bound_dists)
        val = min_d
        grad = np.zeros_like(points)
        # Gradient of boundary min
        i_min = np.argmin(bound_dists)
        if rad[i_min] > 1e-6:
            grad[i_min] -= points[i_min] / rad[i_min]
        return val, grad

    # Construct the list of relevant distances
    # We treat each (i, j) pair where j is a neighbor of i as a constraint.
    # To avoid double counting and ensure symmetry, we can just sum over all i and their neighbors j.
    # Soft min of a set of values d_1...d_m is approx -1/k * log(sum(exp(-k*d_i)))
    
    # We will accumulate the value and gradient for the soft-min of all these local distances.
    # Note: This is a local approximation. The global min is min(all pair dists, all bound dists).
    # The soft min of the *union* of all local neighborhoods approximates the global min well enough for gradient ascent.
    
    all_dists = []
    # Add boundary distances
    all_dists.extend(bound_dists)
    
    # Add pair distances from KDTree
    # We only take unique pairs to avoid double weighting, or just accept the weighting.
    # For gradient ascent, double weighting just scales the gradient, which is fine if k_smooth is fixed.
    # However, to be safe and efficient, we'll just collect distances.
    
    pair_dists = dists[:, 1:] # exclude self
    all_dists.extend(pair_dists.flatten())
    
    all_dists = np.array(all_dists)
    
    # Filter out very small/invalid distances if any (shouldn't happen with KDTree)
    all_dists = all_dists[all_dists > 1e-12]
    
    if len(all_dists) == 0:
        return 0.0, np.zeros_like(points)
        
    d_min = np.min(all_dists)
    # Soft min: d_min - (1/k) * log(sum(exp(-k*(d_i - d_min))))
    # To avoid overflow, we use the log-sum-exp trick.
    # val = d_min - (1/k) * log( sum(exp(-k*(d_i - d_min))) )
    
    exp_terms = np.exp(-k_smooth * (all_dists - d_min))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return d_min, np.zeros_like(points)
        
    soft_val = d_min - (1.0 / k_smooth) * np.log(sum_exp)
    
    # Gradient Calculation
    # d(soft_val) / d(points)
    # Let S = sum(exp(-k*(d_i - d_min)))
    # soft_val = d_min - (1/k) log S
    # Note: d_min is the min of all_dists. If the min changes, the derivative is discontinuous.
    # However, in the smooth region where one distance is strictly the min, 
    # d(soft_val)/d(d_i) = - (1/k) * (exp(-k(d_i - d_min)) / S)
    # Wait, the derivative of - (1/k) log S w.r.t d_i is:
    # - (1/k) * (1/S) * d(S)/d(d_i)
    # d(S)/d(d_i) = exp(-k(d_i - d_min)) * (-k)
    # So term is: - (1/k) * (1/S) * (-k * exp(...)) = exp(...) / S
    # So d(soft_val)/d(d_i) = exp(-k(d_i - d_min)) / S
    
    # We need d(d_i)/d(points).
    # For boundary distance d_i = 1 - |p_i|:
    #   d(d_i)/d(p_i) = - p_i / |p_i|
    # For pair distance d_ij = |p_i - p_j|:
    #   d(d_ij)/d(p_i) = (p_i - p_j) / |p_i - p_j|
    #   d(d_ij)/d(p_j) = (p_j - p_i) / |p_i - p_j|
    
    grad = np.zeros_like(points)
    
    # We need to map the `all_dists` back to their gradients.
    # This is tricky because we flattened them.
    # Let's re-calculate the weights and apply them directly.
    
    # Weights for boundary
    w_bound = np.exp(-k_smooth * (bound_dists - d_min)) / sum_exp
    # grad contribution from boundary
    for i in range(n):
        if rad[i] > 1e-6:
            grad[i] -= w_bound[i] * (points[i] / rad[i])
            
    # Weights for pairs
    # pair_dists shape (n, k_nn)
    # We need to iterate or vectorize.
    # Since k_nn is small (10), we can loop over neighbors for each point.
    
    for i in range(n):
        for k_idx in range(k_nn):
            d_val = dists[i, k_idx + 1]
            j_idx = idxs[i, k_idx + 1]
            
            if d_val < 1e-12:
                continue
                
            w_pair = np.exp(-k_smooth * (d_val - d_min)) / sum_exp
            
            if w_pair < 1e-10:
                continue
                
            diff_vec = points[i] - points[j_idx]
            grad_vec = diff_vec / d_val
            
            grad[i] += w_pair * grad_vec
            grad[j_idx] -= w_pair * grad_vec
            
    return soft_val, grad

def local_optimize(points, time_limit=10.0):
    """
    Perform local optimization using gradient ascent on the soft-min objective.
    """
    start_time = time.time()
    current_points = points.copy()
    current_val, current_grad = soft_min_value_and_grad(current_points, k_smooth=50.0)
    
    best_points = current_points.copy()
    best_val = current_val
    
    # Adaptive learning rate
    lr = 0.01
    k_smooth = 50.0
    
    while time.time() - start_time < time_limit:
        val, grad = soft_min_value_and_grad(current_points, k_smooth=k_smooth)
        
        if val > best_val:
            best_val = val
            best_points = current_points.copy()
            
        # Update
        # Normalize gradient to prevent huge steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
            
        current_points = current_points + lr * grad
        
        # Project back into unit circle
        rad = np.linalg.norm(current_points, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            current_points[mask] /= rad[mask, np.newaxis]
            
        # Occasionally adjust learning rate or k_smooth
        # If we are stuck (small grad), reduce k_smooth to make it smoother? 
        # Or increase lr?
        
        # Simple schedule: decrease lr slowly
        lr *= 0.999
        
    return best_points, best_val

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initialization: Hexagonal Lattice
    # Area of unit circle = pi. N=300. Area per circle ~ pi/300.
    # Hex packing density ~ 0.9069.
    # Effective area per point ~ pi / (300 * 0.9069)
    # Side of hexagon unit ~ sqrt(2 * area / sqrt(3))
    
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    s_gen = s_init * 0.95
    
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 0.99:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_to_center)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
        points = points[:N]
        
    # Multi-start optimization
    # We have ~115s. Let's do 5 restarts of ~20s each.
    # Or 10 restarts of ~10s each.
    num_restarts = 10
    time_per_restart = (time_limit - (time.time() - start_time)) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Add noise to initial points
        noise_scale = 0.05 if r == 0 else 0.02
        current_init = points + np.random.normal(0, noise_scale, size=points.shape)
        
        # Project inside
        rad = np.linalg.norm(current_init, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            current_init[mask] /= rad[mask, np.newaxis]
            
        # Optimize
        opt_points, opt_score = local_optimize(current_init, time_limit=time_per_restart)
        
        # Final precise score calculation
        final_score = compute_score(opt_points)
        
        if final_score > best_score:
            best_score = final_score
            best_points = opt_points
            
    # Final polish on the best result found
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0 and best_points is not None:
        # Do a very short, high-precision polish
        # Use smaller k_smooth for sharper min
        # Re-run local optimize with tighter time budget and maybe higher k_smooth
        polished, pol_score = local_optimize(best_points, time_limit=min(remaining * 0.8, 5.0))
        final_pol_score = compute_score(polished)
        
        if final_pol_score > best_score:
            best_score = final_pol_score
            best_points = polished
            
    # Validation
    if best_points is None:
        return np.zeros((N, 2))
        
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points
