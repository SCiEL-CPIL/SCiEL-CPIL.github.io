"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Construct centres of 200 equal circles in the unit square [-0.5, 0.5]^2
    to maximize the minimum distance between centres and to the boundaries.
    
    Uses a two-stage approach:
    1. Initialize with a hexagonal packing pattern (better than square grid)
    2. Refine using gradient-free optimization (Nelder-Mead) on a subset of points
       to improve the minimum distance.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|))
    We want to maximize this value.
    """
    
    def compute_radius(points):
        """Compute the radius for a given set of points."""
        n = len(points)
        if n == 0:
            return 0.0
        
        # Distance to boundaries
        # Points are in [-0.5, 0.5]^2
        dist_to_boundary = np.minimum(
            np.minimum(0.5 - points[:, 0], points[:, 0] + 0.5),
            np.minimum(0.5 - points[:, 1], points[:, 1] + 0.5)
        )
        min_boundary = np.min(dist_to_boundary)
        
        # Pairwise distances
        # Compute all pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(points, metric='euclidean')
        min_pair = np.min(d) if len(d) > 0 else np.inf
        
        return min(min_boundary, min_pair / 2.0)
    
    def hexagonal_init(n):
        """
        Generate initial positions using a hexagonal close-packing pattern.
        This is denser than a square grid.
        """
        # For n points, estimate the number of rows in a hexagonal arrangement
        # In a hexagonal packing, each row has either k or k-1 points
        # Let's try to find a good arrangement
        
        # Start with a square grid and then shift every other row
        m = int(np.ceil(np.sqrt(n)))
        
        # Create a square grid first
        x = np.linspace(-0.5 + 1.0/(2*m), 0.5 - 1.0/(2*m), m)
        y = np.linspace(-0.5 + 1.0/(2*m), 0.5 - 1.0/(2*m), m)
        
        points = []
        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                if len(points) >= n:
                    break
                # Shift every other row by half the spacing
                dx = 1.0 / m
                if i % 2 == 1:
                    xj_shifted = xj + dx / 2
                    # Keep within bounds
                    if xj_shifted > 0.5 - dx/2:
                        xj_shifted = xj - dx / 2
                points.append([xj_shifted, yi])
        
        points = np.array(points[:n])
        
        # Clamp to be strictly inside
        eps = 1e-10
        points[:, 0] = np.clip(points[:, 0], -0.5 + eps, 0.5 - eps)
        points[:, 1] = np.clip(points[:, 1], -0.5 + eps, 0.5 - eps)
        
        return points
    
    def objective(params):
        """Objective function to minimize (negative of radius)."""
        n = N
        pts = params.reshape(-1, 2)
        # Penalty for going out of bounds
        out_of_bounds = np.sum(
            np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-6) +
            np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-6)
        )
        if out_of_bounds > 1e-4:
            return 100.0 + out_of_bounds
        
        r = compute_radius(pts)
        return -r
    
    # Initialize with hexagonal pattern
    points = hexagonal_init(N)
    
    # Try to improve using local optimization on a subset
    # Since full N=200 is too large for direct optimization,
    # we'll do a few rounds of local refinement
    
    # Simple iterative improvement: for each point, try small perturbations
    # to improve the minimum distance
    
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # Do a few passes of local optimization
    for iteration in range(50):
        improved = False
        for i in range(N):
            original = best_points[i].copy()
            # Try small random perturbations
            for trial in range(20):
                noise = np.random.normal(0, 0.001, 2)
                candidate = best_points.copy()
                candidate[i] = original + noise
                
                # Clamp to bounds
                candidate[i, 0] = np.clip(candidate[i, 0], -0.5 + 1e-10, 0.5 - 1e-10)
                candidate[i, 1] = np.clip(candidate[i, 1], -0.5 + 1e-10, 0.5 - 1e-10)
                
                r_candidate = compute_radius(candidate)
                if r_candidate > best_r + 1e-10:
                    best_points = candidate
                    best_r = r_candidate
                    improved = True
                    break
            if not improved and trial == 0:
                # No improvement in this point, move to next
                pass
        
        if not improved:
            break
    
    # Final clamp to ensure validity
    eps = 1e-10
    best_points[:, 0] = np.clip(best_points[:, 0], -0.5 + eps, 0.5 - eps)
    best_points[:, 1] = np.clip(best_points[:, 1], -0.5 + eps, 0.5 - eps)
    
    return best_points

# EVOLVE-BLOCK-END
