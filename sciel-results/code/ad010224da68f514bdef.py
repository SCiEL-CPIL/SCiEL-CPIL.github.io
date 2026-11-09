"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # 1. Initialization: 3D Fibonacci Spiral
    # This creates points distributed evenly in a sphere of radius 0.5,
    # which fits inside the cube [-0.5, 0.5]^3.
    def fibonacci_spiral(n):
        points = np.zeros((n, 3))
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = (i / (n - 1)) - 0.5  # z from -0.5 to 0.5
            # Clamp z to avoid sqrt of negative due to float error
            z_clamped = np.clip(z, -0.499999, 0.499999)
            r = np.sqrt(1 - z_clamped**2) * 0.5  # Radius of the circle at height z, scaled to 0.5
            theta = 2 * np.pi * i * phi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points[i] = [x, y, z]
        return points

    def soft_min(values, k):
        """
        Calculate the soft minimum of a set of values.
        As k -> infinity, this approaches the hard minimum.
        """
        # Shift values to avoid underflow in exp
        # We want min(v), so we use -1/k * log( mean( exp(-k * v) ) )
        # If values are large, -k*v is very negative, exp is 0.
        # To be numerically stable, we can subtract the min first, but then we lose the softness relative to the min.
        # Actually, the standard formula is:
        # m_k(v) = -1/k * log( (1/n) * sum( exp(-k * v_i) ) )
        
        # To prevent underflow, we can use the log-sum-exp trick.
        # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        # Here x_i = -k * v_i.
        
        scaled = -k * values
        max_val = np.max(scaled)
        # If max_val is very large positive, exp(max_val) might overflow? No, scaled is negative or small.
        # Wait, v_i are positive (distances). So -k * v_i is negative.
        # So max_val is the least negative number (closest to 0).
        
        # log_sum_exp(scaled) = max_val + log(sum(exp(scaled - max_val)))
        log_sum = max_val + np.log(np.sum(np.exp(scaled - max_val)))
        n_vals = len(values)
        # log(mean(exp(scaled))) = log_sum - log(n_vals)
        result = -1.0 / k * (log_sum - np.log(n_vals))
        return result

    def objective(params, k):
        centers = params.reshape(-1, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Get upper triangle indices
        i_upper, j_upper = np.triu_indices(N, k=1)
        pair_dists = dists[i_upper, j_upper] / 2.0
        
        # Boundary clearances
        # For each point, distance to each of the 6 faces
        # Faces are at x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
        # dist to x=0.5 is 0.5 - x
        # dist to x=-0.5 is 0.5 + x
        # So min(0.5 - x, 0.5 + x) = 0.5 - |x|
        # We need the min over all points and all axes.
        abs_centers = np.abs(centers)
        clearances = 0.5 - abs_centers  # shape (N, 3)
        boundary_dists = clearances.flatten()
        
        # Combine all constraints
        all_vals = np.concatenate([pair_dists, boundary_dists])
        
        # We want to maximize the minimum of all_vals.
        # So we minimize -soft_min(all_vals, k).
        m = soft_min(all_vals, k)
        return -m

    # Initial centers
    centers_init = fibonacci_spiral(N)
    params_init = centers_init.flatten()
    
    # Optimization with annealing of k
    # Start with small k to smooth the landscape, increase k to sharpen
    current_params = params_init
    best_score = -np.inf
    
    # A few iterations of increasing k
    k_values = [5, 10, 20, 50, 100]
    
    for k in k_values:
        # Use L-BFGS-B
        # Bounds: centers must be in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Max iterations should be reasonable to fit in time
        result = minimize(
            objective, 
            current_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
        )
        
        if result.success or result.fun < -best_score:
            current_params = result.x
            # Calculate actual hard min score for tracking
            centers = current_params.reshape(-1, 3)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            i_upper, j_upper = np.triu_indices(N, k=1)
            min_pair = np.min(dists[i_upper, j_upper]) / 2.0
            min_bound = np.min(0.5 - np.abs(centers))
            score = min(min_pair, min_bound)
            if score > best_score:
                best_score = score
    
    centers = current_params.reshape(-1, 3)
    
    # Final clip to ensure validity
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
