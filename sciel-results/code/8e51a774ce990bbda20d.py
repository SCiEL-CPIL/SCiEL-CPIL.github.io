import numpy as np
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Initialize with Fibonacci spiral for a good starting distribution
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create a high-resolution grid of test points on the sphere
    # We use a Fibonacci spiral for the grid as well, with more points
    M = 20000
    j = np.arange(M) + 0.5
    z_grid = 1.0 - 2.0 * j / M
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    Y = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Pre-compute dot products? No, P changes.
    
    # Optimization loop
    # We minimize the smooth maximum of the angular distances from Y to nearest P.
    # Using a simple iterative gradient descent with adaptive step size.
    
    # To speed up, we can subsample the grid in later iterations if needed, but 20k x 130 is manageable.
    
    alpha = 0.01
    for it in range(5000):
        # Compute distances from each Y_j to all P_i
        # D[j, i] = arccos(Y_j . P_i)
        # To avoid arccos instability, we can work with cosines or just use the vectors.
        # Let's compute the nearest neighbor index and distance for each Y_j.
        
        # Dot products: (M, N)
        dots = Y @ P.T
        # Clip to [-1, 1] for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        
        # Find nearest P for each Y (max dot product)
        nearest_idx = np.argmax(dots, axis=1)
        nearest_dots = dots[np.arange(M), nearest_idx]
        
        # Angular distance
        dists = np.arccos(nearest_dots)
        
        # Smooth max approximation (Log-Sum-Exp)
        # To prevent overflow, subtract max
        max_dist = np.max(dists)
        # Use a temperature T that decreases over time for sharper focus
        T = 0.05 * (1.0 - it / 5000.0) + 0.01
        exp_dists = np.exp((dists - max_dist) / T)
        sum_exp = np.sum(exp_dists)
        weights = exp_dists / sum_exp
        
        # Calculate gradient for each P_i
        # The gradient of the distance d(y, p) with respect to p on the sphere
        # is in the direction of the projection of (y - p) onto the tangent plane at p.
        # Specifically, grad_p d = - (y - p) / ||y - p||_tangent ?
        # Actually, if we move p towards y, the distance decreases.
        # So we want to move P_i towards the weighted average of Y_j assigned to it.
        
        # Accumulate weighted Y vectors for each P_i
        # We need to project the direction onto the tangent space of P_i.
        
        # Efficient way:
        # For each i, sum of weights * Y_j for all j where nearest_idx == i
        # Then project this sum onto the tangent plane at P_i.
        
        # Scatter-add
        weighted_Y_sum = np.zeros((N, 3))
        np.add.at(weighted_Y_sum, nearest_idx, weights[:, None] * Y)
        
        # For each P_i, the direction to move is the projection of weighted_Y_sum[i] onto the tangent plane at P_i.
        # v_i = weighted_Y_sum[i] - P_i * (P_i . weighted_Y_sum[i])
        
        dots_sum = np.sum(P * weighted_Y_sum, axis=1)
        gradients = weighted_Y_sum - P * dots_sum[:, None]
        
        # Normalize gradients to have similar magnitude?
        # The magnitude of the gradient depends on the number of points.
        # We can normalize by the sum of weights for that point.
        weight_sums = np.zeros(N)
        np.add.at(weight_sums, nearest_idx, weights)
        
        # Avoid division by zero
        mask = weight_sums > 1e-6
        gradients[mask] /= weight_sums[mask, None]
        
        # Update P
        # P_new = P + alpha * gradients
        # Then normalize
        P_new = P + alpha * gradients
        P = P_new / np.linalg.norm(P_new, axis=1, keepdims=True)
        
        # Decay step size
        alpha *= 0.999
        
        # Optional: Restart or add noise if stuck?
        # For now, keep it simple.
        
    return P
# EVOLVE-BLOCK-END
