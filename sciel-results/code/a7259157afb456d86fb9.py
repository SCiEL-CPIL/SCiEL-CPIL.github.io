import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Minimum distance between centers
    # Use numpy broadcasting for N=50 (fast enough)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary
    r = min(min_dist / 2.0, min_boundary)
    
    return r if r > 0 else 0.0

def initial_rings():
    """
    Generates an initial set of points using concentric rings.
    This is a much better structural prior than a random or hexagonal cut.
    """
    points = np.zeros((N, 2))
    # Place first point at center
    points[0] = [0, 0]
    count = 1
    
    # Define ring radii and capacities
    # Optimal packing often has rings of 6, 12, 18, 14 (total 51, so 50 is close)
    # Radii roughly: 0, 0.2, 0.4, 0.6, 0.8
    # Let's use a heuristic distribution.
    
    # We will fill rings starting from inside out.
    # Ring 1: radius ~ 0.25, 6 points
    # Ring 2: radius ~ 0.5, 12 points
    # Ring 3: radius ~ 0.75, 18 points
    # Ring 4: radius ~ 0.95, 14 points
    # Total: 1 + 6 + 12 + 18 + 14 = 51. We need 50.
    
    rings = [
        (0.25, 6),
        (0.50, 12),
        (0.75, 18),
        (0.95, 14) # We will only take 13 if needed
    ]
    
    for radius, capacity in rings:
        for k in range(capacity):
            if count >= N:
                break
            angle = 2 * np.pi * k / capacity
            # Add small perturbation to break symmetry
            noise = np.random.normal(0, 0.01, 2)
            points[count] = [radius * np.cos(angle), radius * np.sin(angle)] + noise
            count += 1
        if count >= N:
            break
            
    return points

def push_apart(points, iterations=20):
    """
    Fast vectorized push-apart algorithm to resolve overlaps.
    This ensures we start the gradient optimization from a feasible (or nearly feasible) state.
    """
    for _ in range(iterations):
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find current min distance
        min_dist = np.min(dists)
        
        # Estimate radius from current config
        # We want to push until min_dist is large.
        # Let's target a radius of 0.12 (approx 0.125)
        target_dist = 2 * 0.12
        
        if min_dist > target_dist:
            # Check boundary
            norms = np.linalg.norm(points, axis=1)
            if np.all(norms <= 1.0):
                break
            # If inside boundary but overlapping, we might be stuck.
            # For now, just continue if overlapping.
            if min_dist > target_dist * 1.1:
                 break

        # Identify pairs that are too close
        # We want to push pairs with dist < target_dist
        mask = dists < target_dist
        
        # Calculate forces
        # Force magnitude proportional to overlap
        # F = (target_dist - dist) * direction
        # We sum forces for each point
        
        # To avoid division by zero, set dists on diagonal to 1
        dists_safe = dists.copy()
        np.fill_diagonal(dists_safe, 1.0)
        
        # Overlap amount
        overlap = np.maximum(0, target_dist - dists)
        overlap = overlap / dists_safe # Normalize direction
        
        # Force vector for each pair is overlap * (p_i - p_j)
        # Actually, diff is p_i - p_j.
        # Force on i is sum over j of overlap_ij * diff_ij
        # Force on j is sum over i of overlap_ji * diff_ji = - Force on i
        
        forces = np.zeros_like(points)
        
        # Vectorized force calculation
        # forces[i] = sum_j overlap[i,j] * (points[i] - points[j])
        # This is equivalent to:
        # forces = (overlap * diff) summed over axis 1?
        # No, diff is (N, N, 2). overlap is (N, N).
        # We want sum over j for each i.
        
        # weighted_diff[i, j, k] = overlap[i, j] * diff[i, j, k]
        weighted_diff = overlap[:, :, np.newaxis] * diff
        forces = np.sum(weighted_diff, axis=1)
        
        # Apply forces with a step size
        # Damping to prevent oscillation
        forces = forces * 0.5
        
        points += forces
        
        # Project back to unit circle if outside
        norms = np.linalg.norm(points, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            points[out_mask] = points[out_mask] / norms[out_mask, np.newaxis]
            
    return points

def smooth_objective_and_grad(x, beta=50.0):
    """
    Objective: Minimize a smooth approximation of the negative packing radius.
    
    We want to maximize r = min(min_dist/2, min_boundary).
    Let's define a potential energy E that is low when r is large.
    
    A common approach is to minimize the "soft-min" of the constraints.
    Let d_ij = ||p_i - p_j||
    Let b_i = 1 - ||p_i||
    
    We want all d_ij > 2r and b_i > r.
    Equivalently, we want to maximize the minimum of {d_ij/2, b_i}.
    
    Let's use a log-sum-exp approximation of the minimum.
    min(z) ≈ -1/beta * log(sum(exp(-beta * z)))
    
    So we want to maximize:
    r_smooth = -1/beta * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * b_i) )
    
    Minimizing -r_smooth is equivalent to minimizing:
    F(x) = 1/beta * log( sum_{i<j} exp(-beta * d_ij/2) + sum_i exp(-beta * b_i) )
    
    Wait, if d_ij is large, exp(-beta * d_ij/2) is small.
    If d_ij is small (overlap), exp(-beta * d_ij/2) is large.
    So F(x) is large when there are overlaps.
    This is a valid energy function to minimize.
    
    However, the gradient of log-sum-exp is a weighted average of the gradients of the terms.
    """
    points = x.reshape(N, 2)
    
    # 1. Boundary terms
    norms = np.linalg.norm(points, axis=1)
    b = 1.0 - norms
    
    # 2. Pairwise terms
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # We only need i < j to avoid double counting
    # But for vectorization, let's compute all and divide by 2?
    # Or just use upper triangle.
    # Let's stick to full matrix for simplicity in gradient, but be careful with weights.
    
    # To avoid overflow in exp, shift by max value?
    # Since we are minimizing, large d gives small exp.
    # The terms that matter are the small d.
    # We can just compute exp directly if beta is not too large.
    # beta=50, d~0.1 -> exp(-2.5) ~ 0.08. d~0.2 -> exp(-5) ~ 0.006.
    # Safe.
    
    # Pairwise energy terms: E_pair = exp(-beta * d / 2)
    # Boundary energy terms: E_bound = exp(-beta * b)
    
    E_pair = np.exp(-beta * d / 2.0)
    np.fill_diagonal(E_pair, 0.0) # Self terms are 0
    
    E_bound = np.exp(-beta * b)
    
    # Total sum
    S = np.sum(E_pair) + np.sum(E_bound)
    
    # Objective F = 1/beta * log(S)
    val = np.log(S) / beta
    
    # Gradient
    # d/dx log(S) = (1/S) * dS/dx
    # dS/dx = sum of d(E_k)/dx
    
    grad = np.zeros_like(x)
    
    # Boundary gradient
    # E_bound_i = exp(-beta * (1 - ||p_i||))
    # d E_bound_i / d p_i = exp(...) * (-beta) * d(1 - ||p_i||)/dp_i
    # d(1 - ||p_i||)/dp_i = - p_i / ||p_i||
    # So d E_bound_i / d p_i = E_bound_i * beta * p_i / ||p_i||
    
    # Note: if p_i is 0, gradient is 0.
    inv_norms = np.zeros_like(norms)
    mask_nonzero = norms > 1e-8
    inv_norms[mask_nonzero] = 1.0 / norms[mask_nonzero]
    
    # grad_bound_i = E_bound_i * beta * p_i * inv_norms_i
    # Shape: (N, 1) * (N, 2)
    grad_bound = (E_bound * beta)[:, np.newaxis] * points * inv_norms[:, np.newaxis]
    
    # Pairwise gradient
    # E_pair_ij = exp(-beta * ||p_i - p_j|| / 2)
    # d E_pair_ij / d p_i = E_pair_ij * (-beta/2) * d(||p_i - p_j||)/dp_i
    # d(||p_i - p_j||)/dp_i = (p_i - p_j) / ||p_i - p_j||
    # So d E_pair_ij / d p_i = E_pair_ij * (-beta/2) * (p_i - p_j) / d_ij
    
    # We sum over all j for each i.
    # grad_pair_i = sum_j E_pair_ij * (-beta/2) * (p_i - p_j) / d_ij
    
    # Vectorized:
    # d_safe to avoid div by 0
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    # Factor for each pair
    # factor_ij = E_pair_ij * (-beta/2) / d_ij
    factor = (E_pair * (-beta / 2.0)) / d_safe
    np.fill_diagonal(factor, 0.0)
    
    # grad_pair = sum_j factor_ij * (p_i - p_j)
    # This is: sum_j factor_ij * p_i - sum_j factor_ij * p_j
    # = p_i * sum_j factor_ij - sum_j (factor_ij * p_j)
    
    # sum_j factor_ij is (factor @ 1)
    row_sums = np.sum(factor, axis=1)
    
    # sum_j (factor_ij * p_j) is (factor @ points)
    # factor is (N, N), points is (N, 2) -> result (N, 2)
    weighted_points = factor @ points
    
    grad_pair = points * row_sums[:, np.newaxis] - weighted_points
    
    # Total gradient
    grad_flat = (grad_bound + grad_pair).flatten()
    
    # We are minimizing F.
    # The objective function returned to scipy is F.
    # The gradient returned is dF/dx.
    
    return val, grad_flat

def construct():
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # Strategy:
    # 1. Generate a good initial guess using rings.
    # 2. Push apart to resolve gross overlaps.
    # 3. Use L-BFGS-B on the smooth objective to refine.
    # 4. If time permits, do a few restarts with perturbed initial guesses.
    
    # Initial Guess
    pts = initial_rings()
    pts = push_apart(pts, iterations=50)
    
    current_r = compute_radius(pts)
    if current_r > best_overall_r:
        best_overall_r = current_r
        best_overall_points = pts.copy()
        
    # Optimization
    # We will run L-BFGS-B multiple times with different beta values or restarts.
    # Beta controls the sharpness of the soft-min.
    # High beta -> closer to true min, but sharper landscape.
    # Low beta -> smoother, but might not find the tight packing.
    
    betas = [20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
            
        # Perturb the current best point to escape local minima?
        # Or start from the pushed rings again?
        # Let's start from the current best points, but add a tiny bit of noise
        # to ensure we don't get stuck in a degenerate minimum if beta changes.
        
        x0 = best_overall_points.flatten() + np.random.normal(0, 0.005, N*2)
        
        # Ensure x0 is valid (inside circle)
        pts_check = x0.reshape(N, 2)
        norms = np.linalg.norm(pts_check, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            pts_check[out_mask] = pts_check[out_mask] / norms[out_mask, np.newaxis]
            x0 = pts_check.flatten()
            
        try:
            res = minimize(
                smooth_objective_and_grad,
                x0,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.status in [1, 2, 3, 4, 5, 6]:
                new_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(new_pts)):
                    new_r = compute_radius(new_pts)
                    if new_r > best_overall_r:
                        best_overall_r = new_r
                        best_overall_points = new_pts.copy()
                        
        except Exception as e:
            pass
            
    # Final Polish with a high beta
    if time.time() < deadline - 10 and best_overall_points is not None:
        x0 = best_overall_points.flatten()
        try:
            res = minimize(
                smooth_objective_and_grad,
                x0,
                args=(500.0,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                new_r = compute_radius(new_pts)
                if new_r > best_overall_r:
                    best_overall_r = new_r
                    best_overall_points = new_pts.copy()
        except Exception:
            pass

    return best_overall_points if best_overall_points is not None else initial_rings()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
