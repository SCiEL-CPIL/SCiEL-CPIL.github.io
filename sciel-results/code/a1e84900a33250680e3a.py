import numpy as np
import time
import sys

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def compute_true_radius(points):
    """Compute the true packing radius for a set of points."""
    if len(points) < 2:
        return np.inf
    # Pairwise distances
    # Using broadcasting for memory efficiency if N is small, but N=300 is fine
    # dist[i, j] = ||points[i] - points[j]||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distance is 0)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def soft_min_energy_and_grad(points, alpha):
    """
    Compute the soft-min energy and gradient.
    Energy = (1/alpha) * ln( sum_{i<j} exp(-alpha * d_ij) + sum_i exp(-alpha * (1 - |c_i|)) )
    We want to minimize this.
    """
    N = len(points)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    
    # Combine all terms into a single sum for the log-sum-exp
    # Terms: exp(-alpha * dist_ij) for i<j and exp(-alpha * boundary_dist_i)
    # To avoid overflow/underflow, we can compute log-sum-exp directly.
    # However, for gradient, we need the weights w_k = exp(-alpha * x_k) / sum(exp(-alpha * x_k))
    
    # Flatten pairwise distances (upper triangle)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dist[i_idx, j_idx]
    
    # All distances to consider
    all_dists = np.concatenate([pair_dists, boundary_dist])
    
    # Log-sum-exp trick for stability
    max_val = np.max(all_dists)
    # Note: we are summing exp(-alpha * d). So max of -alpha*d is -alpha*min(d).
    # Let z = -alpha * all_dists
    z = -alpha * all_dists
    z_max = np.max(z)
    
    # exp(z - z_max)
    exp_z = np.exp(z - z_max)
    sum_exp = np.sum(exp_z)
    
    # Energy
    energy = (z_max + np.log(sum_exp)) / alpha
    
    # Weights for gradient
    weights = exp_z / sum_exp
    
    # Gradient calculation
    # d/dx_i [ (1/alpha) ln( sum_k exp(-alpha x_k) ) ]
    # = (1/alpha) * ( sum_k exp(-alpha x_k) * (-alpha) * dx_k/dx_i ) / sum_k exp(-alpha x_k)
    # = - sum_k weights_k * dx_k/dx_i
    
    grad = np.zeros_like(points)
    
    # 1. Pairwise distances contribution
    # For each pair (i, j), x_k = d_ij.
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    # We have weights for each pair in the first N*(N-1)/2 elements of weights.
    pair_weights = weights[:len(pair_dists)]
    
    # Accumulate gradients for points
    # Vectorized accumulation is tricky with variable indices, so we loop or use scatter.
    # N=300, pairs ~ 45k. Looping 45k times in Python is slow.
    # Let's use a more vectorized approach or accept a loop if it's fast enough?
    # 45k iterations * simple ops might take ~0.1s. That's too slow for many steps.
    
    # Alternative: Use np.add.at
    # grad_i += w * (c_i - c_j) / d
    # grad_j += w * (c_j - c_i) / d
    
    # Let's compute the vectors
    # diff_ij = c_i - c_j
    diff_pairs = points[i_idx] - points[j_idx]
    # dist is already computed, but we need to be careful with zeros (should not be 0)
    inv_d = 1.0 / pair_dists
    
    # term = w * diff_pairs / d
    # For i: + term
    # For j: - term
    terms = (pair_weights * inv_d)[:, np.newaxis] * diff_pairs
    
    np.add.at(grad, i_idx, terms)
    np.add.at(grad, j_idx, -terms)
    
    # 2. Boundary distances contribution
    # x_k = 1 - |c_i|
    # d(1 - |c_i|)/d(c_i) = - c_i / |c_i|
    boundary_weights = weights[len(pair_dists):]
    
    # grad_i += - w * ( - c_i / |c_i| ) = w * c_i / |c_i|
    # Note: if |c_i| is 0, gradient is 0.
    radii_safe = np.where(radii > 1e-10, radii, 1.0)
    boundary_grad_terms = (boundary_weights / radii_safe)[:, np.newaxis] * points
    
    grad += boundary_grad_terms
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing
    # Area = pi, N circles, each area ~ pi*r^2. 
    # Effective area per circle ~ 2.31 * r^2 (hex packing density 0.9069)
    # r ~ sqrt(pi / (N * 2.31))
    r_est = np.sqrt(np.pi / (N * 2.31))
    spacing = 2 * r_est * 1.05 # Slightly larger to ensure fit
    
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points_init = np.array(rows)
    radius_init = np.hypot(points_init[:, 0], points_init[:, 1])
    mask = radius_init <= 1.0
    points = points_init[mask]
    
    # If we have more than N, take the ones closest to center? Or random?
    # Taking closest to center is a good heuristic to keep them compact.
    if len(points) > N:
        dist_to_center = np.linalg.norm(points, axis=1)
        points = points[np.argsort(dist_to_center)[:N]]
    elif len(points) < N:
        # Fallback: Random initialization
        points = np.random.uniform(-0.9, 0.9, (N, 2))
        # Project inside circle
        radii = np.linalg.norm(points, axis=1)
        mask = radii > 1.0
        points[mask] = points[mask] / radii[mask, np.newaxis] * 0.9

    best_points = points.copy()
    best_radius = compute_true_radius(best_points)
    
    # 2. Optimization Loop
    # We will run gradient descent with increasing alpha (sharpness)
    alphas = [5, 20, 50, 100, 200]
    
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
            
        # Define objective and gradient for this alpha
        def obj_grad(x):
            pts = x.reshape(-1, 2)
            e, g = soft_min_energy_and_grad(pts, alpha)
            return e, g.flatten()
        
        # Use L-BFGS-B
        # Initial point is the current best points
        x0 = best_points.flatten()
        
        try:
            res = minimize(
                obj_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            new_points = res.x.reshape(-1, 2)
            new_radius = compute_true_radius(new_points)
            
            if new_radius > best_radius:
                best_points = new_points
                best_radius = new_radius
                
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 5:
            break

    # 3. Local Polish with Nelder-Mead on a subset or full?
    # Nelder-Mead is slow for 600 vars.
    # Instead, do a few steps of simple gradient descent with a very high alpha
    # or just return best_points.
    
    # One more refinement: try to "push" points that are close to the boundary or each other
    # using a very high alpha gradient step manually.
    
    if time.time() < deadline - 2:
        alpha_final = 500
        for step in range(100):
            if time.time() > deadline - 2:
                break
            e, g = soft_min_energy_and_grad(best_points, alpha_final)
            # Simple gradient descent step
            # Adaptive step size
            step_size = 0.01
            new_points = best_points - step_size * g
            # Ensure they don't fly away
            radii = np.linalg.norm(new_points, axis=1)
            mask = radii > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / radii[mask, np.newaxis] * 0.99
                
            new_radius = compute_true_radius(new_points)
            if new_radius > best_radius:
                best_points = new_points
                best_radius = new_radius
            else:
                # Reduce step size if it fails
                step_size *= 0.5
                if step_size < 1e-6:
                    break

    return best_points
# EVOLVE-BLOCK-END
