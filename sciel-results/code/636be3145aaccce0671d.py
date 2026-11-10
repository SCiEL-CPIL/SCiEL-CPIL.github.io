import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = time.time() + 140.0  # 10s safety margin

def _hex_lattice_init(n_points):
    """
    Generate initial positions using a hexagonal lattice.
    This provides a much better starting point than a square grid.
    """
    # Estimate spacing based on area
    # Area of hex unit cell is (sqrt(3)/2) * s^2
    # Total area = 1.0
    # n_points * (sqrt(3)/2) * s^2 = 1.0
    s = np.sqrt(2.0 / (n_points * np.sqrt(3)))
    
    # Generate a larger grid to ensure we have enough points
    # Rows are offset by s/2 in x
    # Vertical distance between rows is s * sqrt(3)/2
    
    rows = int(np.ceil(np.sqrt(n_points / 0.866))) + 4
    cols = int(np.ceil(np.sqrt(n_points))) + 4
    
    points = []
    for i in range(-rows, rows + 1):
        for j in range(-cols, cols + 1):
            x = j * s + (i % 2) * (s / 2.0)
            y = i * s * np.sqrt(3) / 2.0
            # Center the grid roughly in the square
            # We will filter and center later
            points.append((x, y))
    
    points = np.array(points)
    
    # Sort by distance from center to pick the most central n_points
    dists = np.sum(points**2, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n_points]]
    
    # Center the points in the unit square [-0.5, 0.5]
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale if necessary to fit? 
    # The spacing s is derived from area, so it should fit.
    # However, boundary effects might make it slightly too large or small.
    # We will let the optimizer adjust.
    # Clip to ensure they start inside
    points = np.clip(points, -0.49, 0.49)
    
    return points

def _energy_and_grad(pos):
    """
    Compute energy and gradient for repulsive potential.
    E = sum_{i<j} 1/d_ij^2
    """
    pos = pos.reshape(-1, 2)
    n = pos.shape[0]
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_d4 = 1 / d^4
    inv_d4 = 1.0 / (dist_sq ** 2)
    np.fill_diagonal(inv_d4, 0.0)
    
    # Energy: 0.5 * sum_{i,j} 1/d^2 (since each pair counted twice in sum)
    # Wait, 1/d^2 is the term. Sum over i<j is 0.5 * sum_{i!=j} 1/d^2.
    # My inv_d4 is 1/d^4. 
    # Let's re-verify the gradient for E = sum_{i<j} d_{ij}^{-2}
    # dE/dx_i = sum_{j != i} d/dx_i ( (x_i-x_j)^2 + (y_i-y_j)^2 )^{-1}
    # = sum_{j != i} -1 * ((x_i-x_j)^2 + (y_i-y_j)^2 )^{-2} * 2(x_i-x_j)
    # = -2 sum_{j != i} (x_i-x_j) / d_{ij}^4
    # 
    # So grad_i = -2 * sum_{j != i} (pos_i - pos_j) * inv_d4[i,j]
    #            = -2 * ( pos_i * sum_{j != i} inv_d4[i,j] - sum_{j != i} pos_j * inv_d4[i,j] )
    
    row_sum = np.sum(inv_d4, axis=1)
    weighted_pos = np.sum(inv_d4[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    
    grad = -2.0 * (pos * row_sum[:, np.newaxis] - weighted_pos)
    
    # Energy value
    energy = 0.5 * np.sum(np.sum(inv_d4 * dist_sq)) # No, E = sum 1/d^2 = sum inv_d4 * dist_sq? No.
    # dist_sq is d^2. inv_d4 is 1/d^4.
    # 1/d^2 = 1/sqrt(dist_sq) ... no.
    # Let's just compute energy directly to be safe.
    # E = 0.5 * sum_{i!=j} 1/sqrt(dist_sq)
    # This is expensive. 
    # Let's use the property that grad is derived from E.
    # If grad is correct, E should be consistent.
    # Let's calculate E = 0.5 * sum_{i,j} 1/d_ij^2
    # 1/d^2 = 1 / dist_sq
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    energy = 0.5 * np.sum(inv_d2)
    
    return energy, grad.flatten()

def _compute_radius(pos):
    """Compute the maximum common radius for given positions."""
    pos = pos.reshape(-1, 2)
    n = pos.shape[0]
    
    # Min distance between any pair
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    # Min distance to wall
    x = pos[:, 0]
    y = pos[:, 1]
    min_x = np.min(0.5 - np.abs(x))
    min_y = np.min(0.5 - np.abs(y))
    min_wall = min(min_x, min_y)
    
    return min(min_dist / 2.0, min_wall)

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initial configuration
    pos0 = _hex_lattice_init(N)
    
    best_pos = pos0.copy()
    best_radius = _compute_radius(best_pos)
    
    # Optimization settings
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try multiple restarts
    # We can afford ~10-20 restarts if each takes <5s
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Perturb the initial position for restarts > 0
        if restart == 0:
            current_pos = pos0.copy()
        else:
            # Add noise to the best position found so far, or to the initial hex lattice?
            # Adding noise to the best position allows escaping local minima.
            # Adding noise to the initial hex lattice explores different basins.
            # Let's mix: 50% from best, 50% from initial with noise.
            if restart % 2 == 0:
                base = best_pos.copy()
                noise_scale = 0.05 * best_radius
            else:
                base = pos0.copy()
                noise_scale = 0.02
                
            np.random.seed(42 + restart * 100)
            current_pos = base + noise_scale * np.random.randn(N, 2)
            current_pos = np.clip(current_pos, -0.49, 0.49)
        
        try:
            result = minimize(
                fun=lambda x: _energy_and_grad(x)[0],
                jac=lambda x: _energy_and_grad(x)[1],
                x0=current_pos.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                new_pos = result.x.reshape(-1, 2)
                # Ensure positions are strictly inside
                new_pos = np.clip(new_pos, -0.5 + 1e-9, 0.5 - 1e-9)
                
                # Check for invalid states (coincident points)
                # If min dist is 0, radius is 0.
                new_radius = _compute_radius(new_pos)
                
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pos = new_pos.copy()
                    
        except Exception:
            pass
        
        # Optional: Local refinement with a different method?
        # L-BFGS-B is usually sufficient.
        
    # Final cleanup
    best_pos = np.clip(best_pos, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pos
