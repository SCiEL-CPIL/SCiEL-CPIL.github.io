import numpy as np
from scipy.optimize import minimize

N = 26

def smooth_max(x, eps=1e-4):
    """Smooth approximation of max(0, x)."""
    return (x + np.sqrt(x**2 + eps)) / 2.0

def objective(params):
    """
    Objective function to minimize.
    params: flat array of length 3*N: [x0, y0, r0, x1, y1, r1, ...]
    """
    # Unpack
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Sum of radii (we minimize negative sum)
    score = -np.sum(r)
    
    penalty = 0.0
    
    # Boundary penalties
    # Distance to nearest wall
    dx = np.minimum(x, 1.0 - x)
    dy = np.minimum(y, 1.0 - y)
    dist_wall = np.minimum(dx, dy)
    wall_viol = r - dist_wall
    penalty += np.sum(smooth_max(wall_viol)**2)
    
    # Overlap penalties
    # Calculate all pairwise distances
    # Use broadcasting for efficiency
    # x, y, r are shape (N,)
    # We need distance matrix D
    # D[i, j] = sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)
    
    # To avoid sqrt for zero distance (i=j), we can just compute for i < j
    # But vectorized is easier.
    # diff_x = x[:, None] - x[None, :]
    # diff_y = y[:, None] - y[None, :]
    # D = np.sqrt(diff_x**2 + diff_y**2)
    
    # r_sum = r[:, None] + r[None, :]
    # viol = r_sum - D
    # penalty += np.sum(smooth_max(viol)**2)
    # Note: This counts each pair twice. It's fine for penalty.
    
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    D = np.sqrt(diff_x**2 + diff_y**2)
    
    r_sum = r[:, None] + r[None, :]
    overlap_viol = r_sum - D
    penalty += np.sum(smooth_max(overlap_viol)**2)
    
    return score + 100.0 * penalty # Weight for penalty

def construct():
    # Initial best solution: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_params = np.array(cells[:N]).flatten()
    best_score = np.sum(best_params[2::3])
    
    # Bounds for optimizer
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Run multiple optimizations with different starts
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        # Generate initial guess
        # Grid with jitter
        side = int(np.ceil(np.sqrt(N)))
        init = []
        for k in range(N):
            gx = (k % side + 0.5) / side
            gy = (k // side + 0.5) / side
            # Add jitter
            gx += rng.uniform(-0.1, 0.1)
            gy += rng.uniform(-0.1, 0.1)
            # Clip to [0,1]
            gx = np.clip(gx, 0.0, 1.0)
            gy = np.clip(gy, 0.0, 1.0)
            # Initial radius
            gr = 0.05
            init.extend([gx, gy, gr])
            
        x0 = np.array(init)
        
        # Optimize
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
        
        if res.success:
            # Check validity
            params = res.x
            x = params[0::3]
            y = params[1::3]
            r = params[2::3]
            
            # Verify constraints strictly
            valid = True
            # Check bounds
            if np.any(r < -1e-6) or np.any(x < -1e-6) or np.any(y < -1e-6) or np.any(x > 1+1e-6) or np.any(y > 1+1e-6):
                valid = False
            if valid:
                # Check wall
                dist_wall = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
                if np.any(r > dist_wall + 1e-6):
                    valid = False
            if valid:
                # Check overlap
                diff_x = x[:, None] - x[None, :]
                diff_y = y[:, None] - y[None, :]
                D = np.sqrt(diff_x**2 + diff_y**2)
                r_sum = r[:, None] + r[None, :]
                # Check if any pair overlaps significantly
                # np.fill_diagonal(r_sum, 0) # Not needed if we check i!=j
                # For i!=j, D[i,j] should be >= r[i]+r[j]
                # We check min(D - r_sum) for i!=j
                # Create mask for i!=j
                mask = ~np.eye(N, dtype=bool)
                min_gap = np.min(D[mask] - r_sum[mask])
                if min_gap < -1e-6:
                    valid = False
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        
    # Format output
    result = best_params.reshape(N, 3)
    return result
