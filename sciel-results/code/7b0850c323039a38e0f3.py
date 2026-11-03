"""Improved packing program for 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize

N = 200


def construct():
    """
    Construct centres of N=200 equal non-overlapping circles inside [-0.5, 0.5]^2
    maximizing the minimum radius.
    
    Uses a grid-based initialization followed by local optimization using
    scipy's L-BFGS-B with a custom objective that maximizes the minimum distance.
    """
    # Start with a hexagonal packing initialization which is denser than square grid
    # For 200 circles, use a hexagonal arrangement
    
    # Hexagonal packing: rows offset by half a spacing
    # We'll try to fit as many as possible in a hex pattern within the square
    # Side length is 1, so from -0.5 to 0.5
    
    # Generate hexagonal lattice points
    # Hexagonal spacing: horizontal distance = d, vertical distance = d * sqrt(3)/2
    # Alternate rows shifted by d/2
    
    # We want to maximize the minimum distance, so let's start with a reasonable estimate
    # For 200 circles in a unit square, area per circle = 1/200, so radius ~ sqrt(1/(4*200*pi)) ~ 0.02
    # But hex packing is denser, so radius can be larger
    
    # Let's use a smarter initialization: hexagonal grid
    # Try different orientations and pick the best
    
    def hex_grid_init(n_circles):
        """Generate hexagonal lattice points centered in the square."""
        points = []
        # Try to fit hexagonal grid
        # Horizontal spacing d, vertical spacing d*sqrt(3)/2
        # For a square of side 1, we can fit approximately:
        # Number of columns ~ 1/d, rows ~ 1/(d*sqrt(3)/2)
        # Total ~ 1/(d^2 * sqrt(3)/2) = 2/(d^2 * sqrt(3))
        # Set equal to n_circles: d = sqrt(2/(n_circles * sqrt(3)))
        
        d = np.sqrt(2.0 / (n_circles * np.sqrt(3)))
        
        # Generate points
        row = 0
        while len(points) < n_circles:
            # Offset for this row
            offset = 0.0 if row % 2 == 0 else d / 2.0
            
            # Number of points in this row
            # x ranges from -0.5 + d/2 to 0.5 - d/2 with spacing d
            x_start = -0.5 + d / 2.0 - offset
            x_end = 0.5 - d / 2.0 - offset
            
            # Adjust to stay within bounds
            x_vals = np.arange(x_start, x_end + d/2, d)
            # Clip to ensure within [-0.5, 0.5]
            x_vals = x_vals[(x_vals >= -0.5) & (x_vals <= 0.5)]
            
            y = -0.5 + d * np.sqrt(3) / 2.0 * row + d * np.sqrt(3) / 4.0
            
            if y > 0.5:
                break
                
            for x in x_vals:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                    if len(points) >= n_circles:
                        break
            
            row += 1
        
        if len(points) >= n_circles:
            return np.array(points[:n_circles])
        else:
            # Fallback to square grid
            m = int(np.ceil(np.sqrt(n_circles)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:n_circles]
    
    # Initial guess
    x0 = hex_grid_init(N)
    
    # Objective function: negative of the minimum radius
    # r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|))
    
    def objective(x):
        centers = x.reshape(-1, 2)
        
        # Check bounds
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            return 1.0  # Large penalty
        
        # Distance to boundaries
        dist_bound = np.minimum(0.5 - np.abs(centers), axis=1)
        min_bound = np.min(dist_bound)
        
        # Pairwise distances
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        r = min(min_bound, min_pair)
        return -r
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimize
    result = minimize(
        objective,
        x0.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    centers = result.x.reshape(-1, 2)
    
    # Ensure centers are within bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
