"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # 1. Generate initial points using the Golden-angle (Fibonacci) spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    p_cart = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Convert to spherical coordinates (theta, phi) for optimization
    theta = np.arccos(z)
    phi_sph = np.arctan2(p_cart[:, 1], p_cart[:, 0])
    
    # Flatten parameters for the optimizer
    x0 = np.concatenate([theta, phi_sph])
    
    # 3. Define the energy function to minimize
    # We use a smooth approximation of the covering radius (max-min distance).
    # This objective penalizes large gaps between points, directly targeting the covering radius.
    
    def energy(params):
        theta = params[:N]
        phi_sph = params[N:]
        
        # Convert spherical to Cartesian
        x = np.sin(theta) * np.cos(phi_sph)
        y = np.sin(theta) * np.sin(phi_sph)
        z = np.cos(theta)
        p = np.column_stack([x, y, z])
        
        # Compute pairwise dot products
        dots = p @ p.T
        # Clamp to [-1, 1] for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        
        # Compute squared distances (chordal distance squared)
        # d^2 = 2 - 2 * dot
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0) # Set diagonal to 1 to avoid 0 distance in log/sum
        
        # For each point i, we want to approximate min_{j != i} d_{ij}
        # Using a softmin: -tau * log(sum(exp(-d2/tau)))
        # We want to maximize the minimum distance, so we minimize the negative of the max softmin?
        # Actually, we want to minimize the covering radius.
        # Covering radius is related to the max over i of the min distance to neighbors?
        # No, covering radius is the max distance from ANY point on sphere to nearest point.
        # This is approximately the max over i of (distance from Pi to farthest point in its Voronoi cell).
        # A good proxy is the max over i of the distance to the nearest neighbor? No, that's packing.
        
        # Let's use the sum of the largest distances to nearest neighbors.
        # Or simply: Minimize the maximum of the "local covering radii".
        
        # Let's stick to a simpler, robust objective:
        # Minimize the sum of the squared distances to the nearest neighbor for each point?
        # This minimizes the packing radius.
        
        # Let's try: Minimize the sum of the largest 10% of the nearest-neighbor distances.
        # This forces the worst-case gaps to be small.
        
        # Compute nearest neighbor distance for each point
        nn_d2 = np.min(d2, axis=1)
        
        # Sort and take the largest 10% (or a fixed number like 10)
        # To make it differentiable, we can use a soft-max over the sorted values?
        # Simpler: Just minimize the max of nn_d2 using a softmax approximation.
        
        # Softmax approximation of max(nn_d2)
        # max(x) ~ tau * log(sum(exp(x/tau)))
        tau = 0.01
        max_nn_d2 = tau * np.log(np.sum(np.exp(nn_d2 / tau)))
        
        # Also add a small Coulomb term to prevent points from collapsing
        # and to break symmetries
        inv_dist = 1.0 / np.sqrt(nn_d2 + 1e-6) # Avoid div by zero
        coulomb_term = np.sum(inv_dist)
        
        # Weighted sum
        return 10.0 * max_nn_d2 + 0.1 * coulomb_term

    # 4. Perform optimization
    res = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # 5. Extract the optimized parameters
    theta_opt = res.x[:N]
    phi_opt = res.x[N:]
    
    # 6. Convert back to Cartesian coordinates
    x = np.sin(theta_opt) * np.cos(phi_opt)
    y = np.sin(theta_opt) * np.sin(phi_opt)
    z = np.cos(theta_opt)
    p_opt = np.column_stack([x, y, z])
    
    # 7. Normalize vectors to ensure they lie exactly on the unit sphere
    norms = np.linalg.norm(p_opt, axis=1, keepdims=True)
    # Avoid division by zero if a norm is 0 (should not happen)
    norms[norms == 0] = 1.0
    p_opt = p_opt / norms
    
    return p_opt
# EVOLVE-BLOCK-END
