"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 12-fold symmetry: 300 / 12 = 25 points per sector
    N_SECTOR = 25
    # Parameters: (rho, theta) for each of the 25 points
    # rho: radial distance, theta: angle in [0, pi/6]
    
    # Initialization: Ring structure
    # Rings 1-5: 1, 2, 3, 4, 5 points per sector
    # Ring 6: 6 points per sector
    # Ring 7: 4 points per sector (partial)
    # Total: 1+2+3+4+5+6+4 = 25
    
    # Approximate radii for rings in a hexagonal packing of unit radius circles
    # r_circle ~ 0.053
    # Ring m radius ~ m * 2 * r_circle * 0.866 (hex spacing)
    # Let's estimate r_circle = 0.053
    # r_m = m * 2 * 0.053 * 0.866 = m * 0.0918
    # R1: 0.092, R2: 0.184, R3: 0.276, R4: 0.368, R5: 0.460, R6: 0.552, R7: 0.644
    
    # We will optimize these radii and angles.
    
    def get_points(params):
        """
        params: array of shape (50,)
        [rho_0, theta_0, rho_1, theta_1, ...]
        Returns: array of shape (300, 2)
        """
        rhos = params[0::2]
        thetas = params[1::2]
        
        # Base points in sector
        base_x = rhos * np.cos(thetas)
        base_y = rhos * np.sin(thetas)
        
        # Generate 12 rotations
        angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        all_x = np.zeros(300)
        all_y = np.zeros(300)
        
        for k in range(12):
            c = np.cos(angles[k])
            s = np.sin(angles[k])
            idx_start = k * 25
            idx_end = (k + 1) * 25
            
            # Rotation matrix applied to base points
            # x' = x*c - y*s
            # y' = x*s + y*c
            all_x[idx_start:idx_end] = base_x * c - base_y * s
            all_y[idx_start:idx_end] = base_x * s + base_y * c
            
        return np.column_stack((all_x, all_y))

    def objective(params):
        """
        Minimize negative soft-min radius.
        """
        points = get_points(params)
        
        # Check bounds
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            # Penalty for being outside
            return 1000.0 + np.sum(np.maximum(0, radii - 1.0))
            
        # Pairwise distances
        # Compute all pairwise distances
        # dist^2 = (x1-x2)^2 + (y1-y2)^2
        # We only need the min distance.
        # Using cdist is efficient
        from scipy.spatial.distance import cdist
        dists = cdist(points, points, 'euclidean')
        
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Min pair distance
        min_pair_dist = np.min(dists)
        
        # Min boundary distance
        min_boundary_dist = np.min(1.0 - radii)
        
        # We want to maximize min(min_pair_dist/2, min_boundary_dist)
        # Let r = min(min_pair_dist/2, min_boundary_dist)
        # We use a soft min to make it differentiable
        # Soft min of a, b: -1/beta * log(exp(-beta*a) + exp(-beta*b))
        
        beta = 50.0  # Sharpness
        a = min_pair_dist / 2.0
        b = min_boundary_dist
        
        # If a or b is very small, the gradient might be noisy
        # Clip to avoid underflow
        a = np.clip(a, 1e-6, None)
        b = np.clip(b, 1e-6, None)
        
        soft_min = -1.0 / beta * np.log(np.exp(-beta * a) + np.exp(-beta * b))
        
        return -soft_min

    # Initial guess
    # Ring radii estimates
    # We'll use a slightly tighter estimate to start
    r_init = 0.052
    ring_radii = []
    for m in range(1, 8):
        ring_radii.append(m * 2 * r_init * np.sqrt(3)/2)
        
    # Distribute 25 points among 7 rings
    counts = [1, 2, 3, 4, 5, 6, 4]
    
    params_init = np.zeros(50)
    idx = 0
    for m, count in enumerate(counts):
        r_ring = ring_radii[m]
        # Spread angles in [0, pi/6]
        # For symmetry, we want them centered?
        # If we have k points in a sector of pi/6, we can place them at:
        # theta_i = (i + 0.5) * (pi/6) / k
        for i in range(count):
            theta = (i + 0.5) * (np.pi / 6.0) / count
            params_init[idx] = r_ring
            params_init[idx + 1] = theta
            idx += 2
            
    # Bounds
    # rho: [0, 1]
    # theta: [0, pi/6]
    bounds = []
    for i in range(25):
        bounds.append((0, 1.0)) # rho
        bounds.append((0, np.pi/6.0)) # theta

    best_score = -np.inf
    best_params = params_init
    
    # Optimization Loop
    # Use L-BFGS-B
    # We might need to restart or use multiple betas
    
    current_params = params_init
    
    # Run optimization
    try:
        res = minimize(objective, current_params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < objective(current_params):
            current_params = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        points = get_points(current_params)
        return points

    # Refine with higher beta?
    # The objective function uses a fixed beta. 
    # We can do a second pass with a higher beta to sharpen the min.
    
    def objective_sharp(params):
        points = get_points(params)
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            return 1000.0 + np.sum(np.maximum(0, radii - 1.0))
        from scipy.spatial.distance import cdist
        dists = cdist(points, points, 'euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_boundary_dist = np.min(1.0 - radii)
        
        beta = 200.0
        a = min_pair_dist / 2.0
        b = min_boundary_dist
        a = np.clip(a, 1e-6, None)
        b = np.clip(b, 1e-6, None)
        soft_min = -1.0 / beta * np.log(np.exp(-beta * a) + np.exp(-beta * b))
        return -soft_min

    try:
        res2 = minimize(objective_sharp, current_params, method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res2.success or res2.fun < objective_sharp(current_params):
            current_params = res2.x
    except Exception:
        pass

    # Final check and return
    points = get_points(current_params)
    
    # Ensure validity: if any point is outside, clip it?
    # The optimizer should keep them inside, but numerical errors might occur.
    # If a point is outside, the score is invalid.
    # We should project points back to the disk if they are slightly outside?
    # No, projection changes the packing.
    # Better to ensure the optimizer respects bounds.
    # L-BFGS-B respects bounds.
    
    # One more check: if any distance is 0 (coincident points), it's invalid.
    # This is unlikely with continuous optimization.
    
    return points

# EVOLVE-BLOCK-END
