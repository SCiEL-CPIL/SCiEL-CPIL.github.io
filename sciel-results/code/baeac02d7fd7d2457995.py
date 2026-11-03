# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _compute_radius(pts):
    """Compute the packing radius for a given set of points."""
    if len(pts) == 0:
        return 0.0
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    
    radii = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - np.max(radii)
    
    return min(min_pair, min_bound)

def _hex_init(n_points, seed=0):
    """
    Generate multiple diverse hexagonal lattice initializations 
    and return the best one based on initial packing radius.
    """
    best_pts = None
    best_r = -1.0
    
    # Try different rotations and slight spacing variations
    for attempt in range(10):
        # Base spacing estimation
        # Area of unit circle is pi. 
        # Area of N circles is N * pi * r^2.
        # Hex density is ~0.9069.
        # N * (2r)^2 * (sqrt(3)/2) * 0.9069 ~ pi
        # r ~ sqrt(pi / (N * 4 * sqrt(3)/2 * 0.9069))
        # Let's just start with r ~ 0.053 and adjust
        base_spacing = 2.0 / (np.sqrt(n_points) * 1.1)
        
        # Vary spacing slightly to find a better fit
        spacing = base_spacing * (0.95 + 0.01 * attempt)
        
        # Generate points in a grid that covers the unit circle
        k = int(1.0 / spacing) + 5
        
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Standard hex lattice
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Apply rotation
                angle = attempt * np.pi / 5.0
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                
                if x_rot**2 + y_rot**2 <= (1.0 - spacing*0.5)**2:
                    points.append([x_rot, y_rot])
        
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        
        # Sort by distance from center to get the innermost N points
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Calculate radius
        r = _compute_radius(pts)
        
        if r > best_r:
            best_r = r
            best_pts = pts

    if best_pts is None:
        # Fallback: Random points scaled to fit
        best_pts = np.random.rand(n_points, 2) * 1.8 - 0.9
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis] * 0.9
        best_r = _compute_radius(best_pts)

    return best_pts

def _objective_and_grad(x, lam):
    """
    Objective: Minimize -soft_min(distances)
    This is equivalent to Maximize soft_min(distances)
    Distances: all pairwise distances and (1 - |c_i|)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # 1. Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # 2. Compute boundary distances
    radii = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - radii
    
    # 3. Combine all distances into a single vector for log-sum-exp
    iu = np.triu_indices(n, 1)
    d_pairs = dist[iu]
    all_dists = np.concatenate([d_pairs, bound_dist])
    
    # 4. Objective Value
    # soft_min = min_val - (1/lam) * log(sum(exp(-lam * (d - min_val))))
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_terms = np.exp(-lam * shifted)
    log_sum = np.log(np.sum(exp_terms))
    m_soft = min_val - log_sum / lam
    obj = -m_soft
    
    # 5. Gradient
    # Grad = sum P_k * d d_k/dx
    # P_k = exp(-lam * (d_k - min_val)) / sum(exp(-lam * (d - min_val)))
    S = np.sum(exp_terms)
    P = exp_terms / S
    
    n_pairs = len(d_pairs)
    P_pairs = P[:n_pairs]
    P_bonds = P[n_pairs:]
    
    # Gradient accumulation
    grad = np.zeros_like(pts)
    
    # A. Boundary Gradient
    # d (1 - |c_i|) / d c_i = - c_i / |c_i|
    radii_safe = np.maximum(radii, 1e-8)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    grad_bound = -P_bonds[:, np.newaxis] * unit_vecs
    
    # B. Pairwise Gradient
    # Create weight matrix W
    W = np.zeros((n, n))
    # Avoid division by zero (dist is inf on diagonal, but we only fill upper triangle)
    # d_pairs is valid
    W[iu[0], iu[1]] = P_pairs / d_pairs
    W[iu[1], iu[0]] = P_pairs / d_pairs
    
    # Grad_pair[i] = sum_j W[i, j] * (c_i - c_j)
    row_sums = np.sum(W, axis=1)
    W_pts = W @ pts
    grad_pair = pts * row_sums[:, np.newaxis] - W_pts
    
    total_grad = grad_bound + grad_pair
    
    return obj, total_grad.ravel()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit is 150s)
    
    # 1. Initialization
    points = _hex_init(N)
    x0 = points.ravel()
    best_x = x0.copy()
    best_r = _compute_radius(points)
    
    # 2. Optimization with Annealing
    # We will run several stages with increasing lambda (sharpness)
    # Start smooth, end sharp
    
    lambdas = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_x = x0.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        # Use L-BFGS-B
        # maxiter scaled by lambda? Smaller lambda needs more iterations to move?
        # Actually, larger lambda makes the problem stiffer, so fewer iterations might suffice for local polish.
        maxiter = 300
        
        try:
            result = minimize(
                _objective_and_grad, 
                current_x, 
                args=(lam,),
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if result.x is not None:
                new_r = _compute_radius(result.x.reshape(-1, 2))
                if new_r > best_r:
                    best_r = new_r
                    best_x = result.x.copy()
                # Always move to the new position if it didn't get worse significantly?
                # Or just keep the best?
                # If we keep the best, we might stagnate if a slightly worse step leads to a better local min.
                # But with annealing, we are sharpening.
                # Let's update current_x to result.x if it's valid, to continue from there.
                current_x = result.x.copy()
                
        except Exception as e:
            # If optimization fails, break
            break
            
    # 3. Final Polish Pass
    # Identify "active" constraints and try to optimize with very high lambda
    if time.time() < deadline - 5:
        pts = best_x.reshape(-1, 2)
        # Calculate distances to find critical points
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        
        radii = np.linalg.norm(pts, axis=1)
        min_bound = 1.0 - np.max(radii)
        
        r_current = min(np.sqrt(min_pair_sq) / 2.0, min_bound)
        
        # Find indices of points involved in min pair or min bound
        # For simplicity, just run a high lambda optimization on the whole set
        # This acts as the "local refinement"
        
        lam_final = 200.0
        try:
            result_final = minimize(
                _objective_and_grad, 
                best_x, 
                args=(lam_final,),
                method='L-BFGS-B', 
                jac=True, 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            if result_final.x is not None:
                final_r = _compute_radius(result_final.x.reshape(-1, 2))
                if final_r > best_r:
                    best_r = final_r
                    best_x = result_final.x.copy()
        except Exception:
            pass

    # Final validity check and clipping
    final_pts = best_x.reshape(-1, 2)
    radii = np.linalg.norm(final_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / radii[mask, np.newaxis] * 0.999999
        
    return final_pts
# EVOLVE-BLOCK-END
