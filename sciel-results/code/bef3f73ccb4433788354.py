"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np
from scipy.optimize import minimize

N = 200


def construct():
    # We use a physics-inspired approach:
    # 1. Start with a good initial configuration (hexagonal-like packing)
    # 2. Use a soft-contact potential to push circles apart
    # 3. Use gradient-based optimization to maximize the minimum distance
    
    # Initial configuration: hexagonal lattice
    # For 200 circles, use a hexagonal arrangement
    # Hexagonal packing has density ~0.9069 vs square 1.0
    # But we need to fit in unit square [-0.5, 0.5]^2
    
    # Let's try a hexagonal grid
    # In hexagonal packing, rows are offset
    # Row spacing: r * sqrt(3)
    # Column spacing within row: 2r
    # Offset rows: r * sqrt(3)/2 vertical, r horizontal shift
    
    # We'll use a different approach: start with a square grid but perturb it
    # then use optimization to improve
    
    # Actually, let's use a smarter initialization:
    # Start with a hexagonal lattice pattern
    
    def hexagonal_init(n_circles):
        """Generate hexagonal lattice points centered in unit square."""
        # Hexagonal packing: 
        # x = i * 2r + (j % 2) * r
        # y = j * r * sqrt(3)
        # We want to fit as many as possible in [-0.5, 0.5]^2
        
        # Let's try to fit a hexagonal grid
        # For a hexagonal grid with radius r, the bounding box is approximately:
        # width = (cols-1)*2r + r = cols*2r - r
        # height = (rows-1)*r*sqrt(3)
        
        # We want to find r, cols, rows such that we have at least n_circles
        # and the grid fits in the square
        
        best_points = None
        best_r = 0
        
        # Try different numbers of rows
        for rows in range(10, 25):
            for cols in range(10, 25):
                total = rows * cols
                if total < n_circles:
                    continue
                
                # Hexagonal grid dimensions
                # Horizontal spacing: 2r
                # Vertical spacing: r * sqrt(3)
                # With offset, the effective width is (cols-1)*2r + r = (2*cols-1)*r
                # Height: (rows-1)*r*sqrt(3)
                
                # Fit in square [-0.5, 0.5]^2
                # (2*cols-1)*r <= 1 => r <= 1/(2*cols-1)
                # (rows-1)*r*sqrt(3) <= 1 => r <= 1/((rows-1)*sqrt(3))
                
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max > best_r:
                    best_r = r_max
                    # Generate points
                    pts = []
                    for j in range(rows):
                        for i in range(cols):
                            x = (i * 2 - (cols - 1)) * r_max  # centered
                            if j % 2 == 1:
                                x += r_max  # offset
                            y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                            pts.append([x, y])
                    pts = np.array(pts[:n_circles])
                    # Check if all points are within bounds
                    if np.all(np.abs(pts) <= 0.5 - r_max * 0.01):
                        best_points = pts
                        best_r = r_max
        
        if best_points is not None:
            return best_points
        
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]
    
    # Start with hexagonal initialization
    centers = hexagonal_init(N)
    
    # Now optimize using a soft-contact potential approach
    # We want to maximize the minimum of:
    #   - pairwise distances / 2
    #   - distance to boundary
    
    # This is equivalent to maximizing r such that:
    #   |c_i - c_j| >= 2r for all i < j
    #   |c_ik| <= 0.5 - r for all i, k
    
    # We'll use a penalty-based optimization
    # Objective: maximize min of all constraints
    
    def soft_contact_potential(centers, r_target):
        """Compute a smooth approximation of the minimum clearance."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        
        # Soft minimum using log-sum-exp trick
        # For pairwise: min(dists/2)
        # For boundary: min(bound_dists)
        
        # Use a smooth min: -log(sum(exp(-x/t))) * t for small t
        t = 0.001
        
        # Pairwise part
        min_pair = np.min(dists) / 2
        # Boundary part
        min_bound = np.min(bound_dists)
        
        # Combined soft min
        # Use: softmin(a, b) = -t * log(exp(-a/t) + exp(-b/t))
        a = min_pair
        b = min_bound
        soft_min = -t * np.log(np.exp(-a / t) + np.exp(-b / t))
        
        return soft_min
    
    # Use Nelder-Mead or L-BFGS-B to optimize
    # We maximize the soft minimum clearance
    
    def objective(x):
        """Negative of the soft minimum clearance (to minimize)."""
        c = x.reshape(-1, 2)
        return -soft_contact_potential(c, 0.036)
    
    # Initial point
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        centers_opt = result.x.reshape(-1, 2)
    except Exception:
        centers_opt = centers
    
    # Clip to ensure within bounds
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    
    # Final verification: compute actual radius
    def compute_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2
        
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)
    
    r_opt = compute_radius(centers_opt)
    r_init = compute_radius(centers)
    
    # Use the better configuration
    if r_opt >= r_init:
        return centers_opt
    else:
        return centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    print(f"Radius: {compute_radius(centers):.10f}")
