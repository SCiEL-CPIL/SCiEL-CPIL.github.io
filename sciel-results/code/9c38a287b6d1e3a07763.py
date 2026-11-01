import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    # Variables: R (1) + 50 * (theta, phi) (100) = 101 variables
    # theta in [0, pi], phi in [0, 2*pi]
    
    def unpack(x):
        R = x[0]
        angles = x[1:]
        thetas = angles[0::2]
        phis = angles[1::2]
        
        # Convert spherical to cartesian
        # x = R sin(theta) cos(phi)
        # y = R sin(theta) sin(phi)
        # z = R cos(theta)
        sin_theta = np.sin(thetas)
        centers = np.zeros((N, 3))
        centers[:, 0] = R * sin_theta * np.cos(phis)
        centers[:, 1] = R * sin_theta * np.sin(phis)
        centers[:, 2] = R * np.cos(thetas)
        return centers

    def objective(x):
        centers = unpack(x)
        R = x[0]
        
        # Check bounds: centers must be in [-0.5, 0.5]
        # Since R is the radius from origin, max coord is R.
        # So we need R <= 0.5.
        # Also, we want to maximize the minimum distance to wall and between spheres.
        
        # Distance to wall: 0.5 - R (since max coord is R)
        wall_dist = 0.5 - R
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Exclude self-distances
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        r = min(min_pair_dist / 2.0, wall_dist)
        
        # We want to maximize r, so minimize -r
        # Add a small penalty if R > 0.5 to keep it bounded, though optimizer should handle it
        if R > 0.5:
            return -(0.5 - (R - 0.5)) # penalty
        return -r

    # Initialization
    # Start with R = 0.25
    # Distribute points uniformly on sphere
    # Use Fibonacci sphere for initial angles
    R_init = 0.25
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    thetas = np.zeros(N)
    phis = np.zeros(N)
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        thetas[i] = np.arccos(cos_theta)
        phis[i] = i * golden_angle
    
    x0 = np.zeros(101)
    x0[0] = R_init
    x0[1::2] = thetas
    x0[2::2] = phis
    
    # Optimize
    # Nelder-Mead is good for non-smooth objectives
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    best_x = result.x
    best_centers = unpack(best_x)
    
    # Optional: Local refinement with 3D coordinates?
    # The spherical constraint might be too restrictive.
    # Let's try a few iterations of 3D gradient ascent from the best spherical result to "break" the spherical constraint if beneficial.
    
    centers = best_centers.copy()
    max_iter = 500
    step_size = 0.01
    best_r = 0.0
    best_centers_local = centers.copy()
    
    for i in range(max_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        abs_centers = np.abs(centers)
        max_coord = np.max(abs_centers, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_pair_dist = np.min(dist)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_centers_local = centers.copy()
            
        # Gradient step
        grad = np.zeros_like(centers)
        
        # Active pairs
        active_pairs = np.argwhere(dist < 2 * (current_r + 0.01))
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        for i_idx, j_idx in active_pairs:
            v = centers[i_idx] - centers[j_idx]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-9:
                v_norm = v / norm_v
                grad[i_idx] += v_norm
                grad[j_idx] -= v_norm
                
        # Active walls
        active_walls = np.where(wall_dists < current_r + 0.01)[0]
        for i_idx in active_walls:
            limiting_coords = np.argmax(abs_centers[i_idx])
            direction = -np.sign(centers[i_idx, limiting_coords])
            grad[i_idx, limiting_coords] += direction
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad /= grad_norm
            centers += step_size * grad
            centers = np.clip(centers, -0.5, 0.5)
            
        step_size *= 0.999
        
    return best_centers_local
