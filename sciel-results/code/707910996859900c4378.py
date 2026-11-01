import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
DEADLINE = 145.0  # Safety margin from 150s

def _compute_radius(centers):
    """Compute the valid radius for a given set of centers."""
    # Boundary constraints
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    
    # Pairwise distances
    # Using cKDTree for efficiency, though N=100 is small
    tree = cKDTree(centers)
    # Query k=2 to get nearest neighbor (excluding self)
    # Actually, pdist is easier for min distance
    from scipy.spatial.distance import pdist
    d = pdist(centers, 'euclidean')
    r_pair = np.min(d) / 2.0
    
    return min(r_wall, r_pair)

def _objective_and_grad(centers, r_target):
    """
    Objective: Sum of squared violations for radius r_target.
    Violations:
    1. Pairwise: max(0, 2*r_target - dist_ij)^2
    2. Wall: max(0, r_target - (0.5 - max(abs(c_ik))))^2
    """
    n = centers.shape[0]
    # Reshape to (n, 1, 3) and (1, n, 3) for broadcasting
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    
    # Pairwise
    diff = c1 - c2
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Pairwise penalty
    # We only care about pairs where dist < 2*r_target
    # Let margin = 2*r_target - dist
    # penalty = margin^2 if margin > 0 else 0
    margin = 2.0 * r_target - dist
    active = margin > 0
    # Gradient of margin w.r.t centers[i] is - (centers[i] - centers[j]) / dist
    # Gradient of penalty w.r.t centers[i] = 2 * margin * (- (centers[i] - centers[j]) / dist)
    # Sum over j
    
    # To avoid division by zero in gradient, use safe dist
    dist_safe = np.where(dist > 1e-12, dist, 1.0)
    
    # Weight for gradient: 2 * margin / dist_safe (only where active)
    w = np.zeros_like(dist)
    w[active] = 2.0 * margin[active] / dist_safe[active]
    
    # grad[i] = sum_j w[i,j] * (centers[i] - centers[j]) ? 
    # Wait, d/dx_i (2r - ||x_i - x_j||)^2 = 2(2r - d) * d(2r-d)/dx_i
    # d(2r-d)/dx_i = - d(||x_i-x_j||)/dx_i = -(x_i-x_j)/||x_i-x_j||
    # So grad_i += 2 * margin * (-(x_i-x_j)/dist)
    # My w above is 2*margin/dist.
    # So grad_i += - w[i,j] * (x_i - x_j)
    
    # Vectorized gradient calculation for pairwise
    # grad[i] = - sum_j w[i,j] * (centers[i] - centers[j])
    # = - sum_j w[i,j] * centers[i] + sum_j w[i,j] * centers[j]
    # = - centers[i] * sum_j w[i,j] + (w @ centers)[i]
    
    sum_w = np.sum(w, axis=1)
    weighted_centers = w @ centers
    grad_pair = - sum_w[:, np.newaxis] * centers + weighted_centers
    
    # Wall penalty
    # limit = 0.5 - r_target
    # violation = max(0, |c_ik| - limit)
    limit = 0.5 - r_target
    abs_c = np.abs(centers)
    excess = np.maximum(0.0, abs_c - limit)
    active_wall = excess > 0
    
    # Penalty wall = sum excess^2
    # Grad wall = 2 * excess * sign(c)
    grad_wall = np.zeros_like(centers)
    grad_wall[active_wall] = 2.0 * excess[active_wall] * np.sign(centers)[active_wall]
    
    total_grad = grad_pair + grad_wall
    
    # Objective value
    obj_pair = np.sum(margin[active]**2)
    obj_wall = np.sum(excess[active_wall]**2)
    obj = obj_pair + obj_wall
    
    return obj, total_grad

def _build_halton_init(n, dim=3):
    """Generate Halton sequence points in [0, 1]^dim."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, dim))
    for i in range(n):
        for j in range(dim):
            points[i, j] = halton(i + 1, j + 2) # Bases 2, 3, 5
    # Map to [-0.5, 0.5]
    return points - 0.5

def construct():
    start_time = time.time()
    
    # Initial guess for radius
    # Volume argument: 100 * (4/3) pi r^3 <= 1 => r <= (3/400 pi)^(-1/3) ~ 0.12
    # Best known is ~0.111
    current_r = 0.10
    
    # Use Halton sequence for better initial spread
    centers = _build_halton_init(N)
    
    # Initial optimization to settle down
    # We will try to increase r gradually
    
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Optimization parameters
    max_iter_opt = 200
    tol = 1e-8
    
    # We iterate increasing r
    # Start with a slightly smaller r to ensure feasibility, then push up
    r_target = best_r * 0.95
    
    while time.time() - start_time < DEADLINE - 5:
        # Try to optimize for current r_target
        # If r_target is too small, it will converge quickly.
        # If it's too large, it will fail to reduce penalty to 0.
        
        # Use L-BFGS-B
        # Note: L-BFGS-B requires bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Flatten centers for scipy
        x0 = centers.flatten()
        
        def obj_func(x):
            c = x.reshape(N, 3)
            val, _ = _objective_and_grad(c, r_target)
            return val
        
        def grad_func(x):
            c = x.reshape(N, 3)
            _, g = _objective_and_grad(c, r_target)
            return g.flatten()
            
        try:
            res = minimize(
                obj_func, 
                x0, 
                jac=grad_func, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter_opt, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            break
            
        centers = res.x.reshape(N, 3)
        
        # Check if we achieved the target radius
        # The penalty should be close to 0 if feasible
        # But due to numerical issues, we check the actual radius
        actual_r = _compute_radius(centers)
        
        if actual_r > best_r:
            best_r = actual_r
            best_centers = centers.copy()
            
        # Check if we can increase r_target
        # If the penalty at the solution is small, we likely achieved r_target
        # Let's check the penalty value
        pen, _ = _objective_and_grad(centers, r_target)
        
        if pen < 1e-6:
            # We achieved r_target, try to increase
            r_target += 0.0005
        else:
            # We failed to achieve r_target
            # Reduce r_target slightly and try again to find the boundary
            # Or, if we are stuck, try a small kick
            r_target -= 0.0002
            
            # If we've been stuck (r_target decreasing or not improving best_r), add kick
            # Simple heuristic: if r_target < best_r, we are definitely stuck
            if r_target < best_r:
                # Perturb
                centers += np.random.normal(0, 0.01, centers.shape)
                centers = np.clip(centers, -0.49, 0.49)
                r_target = best_r + 0.001
                # Re-optimize from perturbed state?
                # The loop will do it.
        
        # Safety: if r_target is getting too high (theoretical max ~0.111), stop pushing
        if r_target > 0.115:
             # Do a final polish at the best known r
             break

    # Final Polish: Use the best centers and try to squeeze a bit more
    # We can run a few more iterations at a very tight tolerance
    # Or just return best_centers
    
    # One more attempt with a slightly higher r if we have time
    if time.time() - start_time < DEADLINE - 2:
        r_target = best_r + 0.0001
        x0 = best_centers.flatten()
        try:
            res = minimize(
                lambda x: _objective_and_grad(x.reshape(N, 3), r_target)[0], 
                x0, 
                jac=lambda x: _objective_and_grad(x.reshape(N, 3), r_target)[1].flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_final = res.x.reshape(N, 3)
            r_final = _compute_radius(c_final)
            if r_final > best_r:
                best_centers = c_final
                best_r = r_final
        except Exception:
            pass

    return best_centers
