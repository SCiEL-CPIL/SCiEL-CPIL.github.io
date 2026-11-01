import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialize with Hexagonal Lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # r ~ sqrt(1/N) ~ 0.057.
    # Hex spacing d = 2r.
    # Let's start with a reasonable spacing and adjust.
    
    # Generate points on a hex grid
    def generate_hex_grid(N_target, r_init):
        spacing = 2.0 * r_init
        # Approximate number of points in a circle of radius R with hex spacing d
        # Area ~ pi R^2. Point density ~ 1 / (d^2 * sqrt(3)/2)
        # N ~ pi R^2 / (d^2 * sqrt(3)/2)
        # R ~ d * sqrt(N * sqrt(3) / (2*pi))
        R_est = spacing * np.sqrt(N_target * np.sqrt(3.0) / (2.0 * np.pi))
        R_est *= 1.1  # Add margin
        
        k = int(np.ceil(R_est / spacing)) + 2
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > R_est:
                continue
            x_offset = (j % 2) * spacing / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if np.hypot(x, y) <= R_est:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((N_target, 2))
            
        # If we have more than N, select N closest to center
        if len(points) > N_target:
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)
            points = points[indices[:N_target]]
        elif len(points) < N_target:
            # Pad with random points inside circle if not enough
            # This shouldn't happen with R_est margin
            while len(points) < N_target:
                angle = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, 1)
                points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
        
        return points

    # Initial guess for radius
    r_guess = 0.05
    points = generate_hex_grid(N, r_guess)
    
    # Ensure points are strictly inside
    # Scale down if any point is outside
    max_rad = np.max(np.hypot(points[:, 0], points[:, 1]))
    if max_rad > 1.0:
        scale = 0.99 / max_rad
        points *= scale

    best_points = points.copy()
    best_score = -np.inf

    # Define objective function
    def objective(x):
        p = x.reshape(-1, 2)
        # Calculate pairwise distances
        # Use broadcasting for efficiency
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # This creates a N x N matrix. 300x300 = 90,000 elements.
        
        # More efficient:
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Boundary distances
        boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
        
        # We want to maximize the minimum of (dists/2) and boundary_dists
        # Soft min approximation
        beta = 100.0  # Sharpness parameter
        
        # Combine all constraints
        # dists/2 for i != j
        # boundary_dists for all i
        
        # Mask diagonal for dists
        np.fill_diagonal(dists, np.inf)
        
        min_dists = dists / 2.0
        # Flatten valid distances
        valid_dists = min_dists[np.triu_indices(N, 1)]
        
        # Combine with boundary
        all_vals = np.concatenate([valid_dists, boundary_dists])
        
        # Soft min
        # -1/beta * log(sum(exp(-beta * vals)))
        # To avoid underflow, subtract min
        m = np.min(all_vals)
        if m < 0:
            return m # If any constraint is violated (negative), return it directly
        exp_vals = np.exp(-beta * (all_vals - m))
        soft_min = m - (1.0/beta) * np.log(np.sum(exp_vals))
        
        return -soft_min

    # Gradient (optional, but L-BFGS-B can approximate or we can provide it)
    # Providing gradient is faster.
    def objective_grad(x):
        p = x.reshape(-1, 2)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        
        boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
        
        beta = 100.0
        
        valid_dists = dists / 2.0
        # We need the gradient of the soft min.
        # Let S = sum(exp(-beta * v_i)). soft_min = m - 1/beta log(S).
        # d(soft_min)/dx = -1/beta * (1/S) * sum(d(exp(-beta v_i))/dx)
        # = -1/beta * (1/S) * sum(-beta exp(-beta v_i) dv_i/dx)
        # = 1/S * sum(exp(-beta v_i) dv_i/dx)
        
        # This is a weighted average of the gradients of the individual terms.
        # Weights w_i = exp(-beta (v_i - m)) / sum(exp(-beta (v_j - m)))
        
        # Calculate weights
        all_vals = np.concatenate([valid_dists, boundary_dists])
        m = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - m))
        S = np.sum(exp_vals)
        weights = exp_vals / S
        
        # Separate weights for distances and boundaries
        n_pairs = len(valid_dists)
        w_dists = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        grad = np.zeros_like(x)
        
        # Gradient of boundary terms:
        # v_i = 1 - ||p_i||
        # dv_i/dp_i = -p_i / ||p_i||
        # Contribution: sum(w_bounds_i * dv_i/dp_i)
        norms = np.linalg.norm(p, axis=1)
        norms[norms == 0] = 1.0
        grad_bounds = -p / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
        grad += grad_bounds
        
        # Gradient of distance terms:
        # v_{ij} = ||p_i - p_j|| / 2
        # dv_{ij}/dp_i = (p_i - p_j) / (2 * ||p_i - p_j||)
        # dv_{ij}/dp_j = (p_j - p_i) / (2 * ||p_i - p_j||)
        
        # We have w_dists corresponding to upper triangle pairs.
        # We need to map them back to indices.
        # This is tricky to vectorize efficiently without loops if we don't store indices.
        # For N=300, a loop over pairs might be acceptable if it's inside the gradient call?
        # 45,000 pairs. 100 iterations. 4.5 million operations. Should be fine.
        
        # However, constructing the full N x N weight matrix is O(N^2) memory.
        # Let's try to compute the gradient of the sum of weighted distances.
        
        # We can reconstruct the full N x N weight matrix W where W[i,j] is the weight for dist(i,j)/2.
        # W is symmetric. W[i,i]=0.
        # The values in valid_dists are from np.triu_indices(N, 1).
        # Let's create a full matrix of weights.
        W_full = np.zeros((N, N))
        i_idx, j_idx = np.triu_indices(N, 1)
        W_full[i_idx, j_idx] = w_dists
        W_full[j_idx, i_idx] = w_dists
        
        # Now, grad_i = sum_j W_full[i,j] * dv_{ij}/dp_i
        # dv_{ij}/dp_i = (p_i - p_j) / (2 * dists[i,j])
        # So grad_i = sum_j W_full[i,j] * (p_i - p_j) / (2 * dists[i,j])
        
        # Vectorized:
        # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        # We need to divide by dists[i,j]
        # inv_dists = 1.0 / (2.0 * dists)
        # np.fill_diagonal(inv_dists, 0)
        
        inv_dists = 1.0 / (2.0 * dists)
        np.fill_diagonal(inv_dists, 0.0)
        
        # Weighted sum of differences
        # term[i,j] = W_full[i,j] * inv_dists[i,j] * (p_i - p_j)
        # grad[i] = sum_j term[i,j]
        
        # diff is (N, N, 2)
        # We multiply W_full (N, N) and inv_dists (N, N) to get coeff (N, N)
        coeff = W_full * inv_dists
        # coeff[i,j] * diff[i,j]
        # Sum over j
        grad_dists = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        grad += grad_dists
        
        return grad

    # Optimization loop
    beta_schedule = [10.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
            
        # Update beta in functions
        # Since beta is captured in closure, we need to redefine or use a mutable container
        # For simplicity, we will just restart the optimizer with a new beta value
        # by redefining the functions? No, that's expensive.
        # We can pass beta as a global or use a class.
        # Let's just use a single beta=100.0 which is usually good enough for polish.
        
        pass

    # Let's just run one optimization with beta=100.0
    # If time permits, run again with beta=200.0
    
    x0 = points.flatten()
    
    # Bounds: [-1.1, 1.1] to allow slight excursion during optimization
    bounds = [(-1.1, 1.1)] * (2 * N)
    
    try:
        res = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        
        if res.success or res.fun < best_score:
            best_points = res.x.reshape(-1, 2)
            best_score = -res.fun
            
            # Second pass with higher beta if time remains
            if time.time() < deadline - 20:
                # Increase beta
                # We need to update the objective/grad to use higher beta.
                # Let's define new functions with beta=200
                beta_high = 200.0
                
                def objective_high(x):
                    p = x.reshape(-1, 2)
                    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
                    
                    np.fill_diagonal(dists, np.inf)
                    min_dists = dists / 2.0
                    valid_dists = min_dists[np.triu_indices(N, 1)]
                    all_vals = np.concatenate([valid_dists, boundary_dists])
                    
                    m = np.min(all_vals)
                    if m < 0:
                        return m
                    exp_vals = np.exp(-beta_high * (all_vals - m))
                    soft_min = m - (1.0/beta_high) * np.log(np.sum(exp_vals))
                    return -soft_min

                def objective_grad_high(x):
                    p = x.reshape(-1, 2)
                    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dists, 1.0)
                    boundary_dists = 1.0 - np.linalg.norm(p, axis=1)
                    
                    valid_dists = dists / 2.0
                    all_vals = np.concatenate([valid_dists, boundary_dists])
                    m = np.min(all_vals)
                    exp_vals = np.exp(-beta_high * (all_vals - m))
                    S = np.sum(exp_vals)
                    weights = exp_vals / S
                    
                    n_pairs = len(valid_dists)
                    w_dists = weights[:n_pairs]
                    w_bounds = weights[n_pairs:]
                    
                    grad = np.zeros_like(x)
                    
                    norms = np.linalg.norm(p, axis=1)
                    norms[norms == 0] = 1.0
                    grad_bounds = -p / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
                    grad += grad_bounds
                    
                    W_full = np.zeros((N, N))
                    i_idx, j_idx = np.triu_indices(N, 1)
                    W_full[i_idx, j_idx] = w_dists
                    W_full[j_idx, i_idx] = w_dists
                    
                    inv_dists = 1.0 / (2.0 * dists)
                    np.fill_diagonal(inv_dists, 0.0)
                    
                    coeff = W_full * inv_dists
                    grad_dists = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
                    
                    grad += grad_dists
                    return grad

                res2 = minimize(objective_high, best_points.flatten(), jac=objective_grad_high, 
                                method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
                
                if res2.success or -res2.fun > best_score:
                    best_points = res2.x.reshape(-1, 2)
                    best_score = -res2.fun

    except Exception as e:
        pass

    # Final check: ensure points are valid (inside circle)
    # If any point is outside, scale down or move inside
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii > 1.0):
        # Simple fix: project to circle boundary or scale
        # Scaling might reduce the radius further, but ensures validity.
        # Better: move the offending points to the boundary in the same direction?
        # Or just scale the whole set slightly?
        max_r = np.max(radii)
        if max_r > 1.0:
            scale = 0.999 / max_r
            best_points *= scale

    return best_points
