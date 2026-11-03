# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_init(n_points, seed=42):
    """
    Generate a hexagonal lattice initialization.
    We generate a large grid of hex points and select the n_points closest to the origin.
    We try a few spacings to ensure we have enough points and to find a good initial packing.
    """
    best_pts = None
    best_r = -np.inf

    # Theoretical optimal radius is ~ 0.05315.
    # Spacing d = 2*r ~ 0.1063.
    # Let's try spacings around this value.
    
    for attempt in range(10):
        # Vary spacing slightly
        # Base estimate: area per point ~ pi/n_points. Hex cell area ~ d^2 * sqrt(3)/2.
        # d ~ sqrt(2 * pi / (n_points * sqrt(3)))
        d_est = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        
        # Adjust for boundary effects (points near edge are wasted)
        # Usually need slightly smaller spacing to fit more points
        # Or just try a range.
        d = d_est * (0.95 + 0.02 * attempt)
        
        # Generate hex lattice
        # x = (i + j/2) * d
        # y = j * d * sqrt(3) / 2
        
        # Range of i, j
        # Max radius 1.
        # j ranges from -1/(d*sqrt(3)/2) to 1/(d*sqrt(3)/2)
        j_max = int(1.0 / (d * np.sqrt(3) / 2.0)) + 2
        i_max = int(1.0 / d) + 2
        
        points = []
        for j in range(-j_max, j_max + 1):
            y = j * d * np.sqrt(3) / 2.0
            if y**2 > 1.0:
                continue
            for i in range(-i_max, i_max + 1):
                x = (i + 0.5 * j) * d
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        
        if len(points) < n_points:
            continue
            
        points = np.array(points)
        
        # Sort by distance from origin
        radii = np.linalg.norm(points, axis=1)
        idx = np.argsort(radii)[:n_points]
        pts = points[idx]
        
        # Check radius
        # Fast check
        d_mat = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d_mat, np.inf)
        min_pair = np.min(d_mat)
        min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
        r = min(min_pair / 2.0, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = pts
            
        # If we are very close to theoretical limit, we can stop
        if r > 0.053:
            break

    if best_pts is None:
        # Fallback: Random
        best_pts = np.random.randn(n_points, 2) * 0.5
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts / norms[:, np.newaxis] * 0.9

    return best_pts

def _compute_radius(pts):
    """Compute the valid packing radius."""
    if len(pts) == 0:
        return 0.0
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, lam):
    """
    Compute -1/lam * log(sum(exp(-lam * values)))
    Approximates min(values).
    """
    min_val = np.min(values)
    shifted = values - min_val
    exp_terms = np.exp(-lam * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / lam

def _objective_and_grad(x, lam):
    """
    Objective: Minimize -soft_min(distances)
    Equivalent to Maximize soft_min(distances).
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq + 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    radii = np.linalg.norm(pts, axis=1)
    bound_dist = 1.0 - radii
    
    # Combine
    iu = np.triu_indices(n, 1)
    d_pairs = dist[iu]
    all_dists = np.concatenate([d_pairs, bound_dist])
    
    # Soft min
    m_soft = _soft_min_logsumexp(all_dists, lam)
    obj = -m_soft
    
    # Gradient
    # Grad = sum P_k * d d_k/dx
    exp_vals = np.exp(-lam * (all_dists - np.min(all_dists)))
    S = np.sum(exp_vals)
    P = exp_vals / S
    
    n_pairs = len(d_pairs)
    P_pairs = P[:n_pairs]
    P_bonds = P[n_pairs:]
    
    grad = np.zeros_like(pts)
    
    # Boundary Gradient
    radii_safe = np.maximum(radii, 1e-8)
    unit_vecs = pts / radii_safe[:, np.newaxis]
    # d (1 - r_i) / d c_i = - c_i / r_i
    # Grad contribution: P_bonds[i] * (- c_i / r_i)
    grad_bound = -P_bonds[:, np.newaxis] * unit_vecs
    
    # Pairwise Gradient
    # d d_ij / d c_i = (c_i - c_j) / d_ij
    # d d_ij / d c_j = (c_j - c_i) / d_ij
    # We need sum over j of P_ij * (c_i - c_j) / d_ij
    
    # Construct weight matrix W
    W = np.zeros((n, n))
    # Avoid division by zero (diagonal is inf, but we only fill upper triangle)
    # d_pairs corresponds to upper triangle
    # We need to handle the case where d_pairs is 0? No, we added 1e-12.
    W[iu[0], iu[1]] = P_pairs / d_pairs
    W[iu[1], iu[0]] = P_pairs / d_pairs
    
    # Grad_pair[i] = sum_j W[i, j] * (c_i - c_j)
    # = c_i * sum_j W[i, j] - sum_j W[i, j] * c_j
    row_sums = np.sum(W, axis=1)
    W_pts = W @ pts
    grad_pair = pts * row_sums[:, np.newaxis] - W_pts
    
    total_grad = grad_bound + grad_pair
    
    return obj, total_grad.ravel()

def _force_relax(pts, max_iters=200, time_limit=10.0):
    """
    Perform local force-directed relaxation.
    This is essentially gradient ascent on the min distance, but using a simpler
    repulsive force model which can be more stable in local minima.
    """
    start_time = time.time()
    current_r = _compute_radius(pts)
    
    # Initial step size
    step = 0.01
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Compute forces
        # Force is repulsive if distance < 2*r_target, attractive otherwise?
        # Actually, we just want to maximize the min distance.
        # Let's use the gradient from _objective_and_grad but with a fixed high lambda?
        # Or a simple heuristic:
        # For each point, sum vectors to neighbors, weighted by 1/d^2 if d < threshold.
        
        # Let's stick to the smooth gradient method but with adaptive lambda.
        # If we are stuck, increase lambda to focus on the tightest constraints.
        
        # Use a moderate lambda for relaxation
        lam = 50.0
        
        obj, grad = _objective_and_grad(pts.ravel(), lam)
        
        # Gradient ascent
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
            
        # Adaptive step size
        # If gradient is large, step smaller?
        # Simple: fixed step scaled by 1/grad_norm? No, that's Newton-like.
        # Let's use a fixed small step.
        
        new_pts = pts + step * grad.reshape(-1, 2)
        
        # Check if new radius is better
        new_r = _compute_radius(new_pts)
        
        if new_r > current_r:
            pts = new_pts
            current_r = new_r
            # Maybe decrease step slightly
            step *= 0.99
        else:
            # Decrease step
            step *= 0.9
            if step < 1e-6:
                break
                
    return pts, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit 150s)
    
    # 1. Initialization
    points = _hex_init(N)
    x0 = points.ravel()
    
    best_x = x0.copy()
    best_r = _compute_radius(points)
    
    # 2. Optimization Stages
    
    # Stage 1: Coarse optimization
    lam1 = 5.0
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    if time.time() < deadline - 30:
        result1 = minimize(
            _objective_and_grad, 
            x0, 
            args=(lam1,),
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        x1 = result1.x
        r1 = _compute_radius(x1.reshape(-1, 2))
        if r1 > best_r:
            best_r = r1
            best_x = x1.copy()
            
    # Stage 2: Medium optimization
    lam2 = 20.0
    if time.time() < deadline - 20:
        result2 = minimize(
            _objective_and_grad, 
            best_x, 
            args=(lam2,),
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        x2 = result2.x
        r2 = _compute_radius(x2.reshape(-1, 2))
        if r2 > best_r:
            best_r = r2
            best_x = x2.copy()
            
    # Stage 3: Fine optimization
    lam3 = 50.0
    if time.time() < deadline - 10:
        result3 = minimize(
            _objective_and_grad, 
            best_x, 
            args=(lam3,),
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        x3 = result3.x
        r3 = _compute_radius(x3.reshape(-1, 2))
        if r3 > best_r:
            best_r = r3
            best_x = x3.copy()
            
    # Stage 4: Local Polish
    # Use force relaxation or high-precision local search
    if time.time() < deadline - 5:
        pts_polished, r_polished = _force_relax(best_x.reshape(-1, 2), max_iters=100, time_limit=5.0)
        if r_polished > best_r:
            best_r = r_polished
            best_x = pts_polished.ravel()
            
    # Final validity check
    final_pts = best_x.reshape(-1, 2)
    radii = np.linalg.norm(final_pts, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        final_pts[mask] = final_pts[mask] / radii[mask, np.newaxis] * 0.999999
        
    return final_pts
# EVOLVE-BLOCK-END
