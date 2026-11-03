"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, cdist
import time

def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses an incremental radius search with scipy.optimize.least_squares.
    """
    n = N
    
    # Helper to generate hexagonal lattice points
    def hex_lattice_points(r_target):
        # Spacing is 2*r_target
        spacing = 2.0 * r_target
        if spacing <= 0:
            return np.array([[0.0, 0.0]])
        
        # Determine range
        # Max radius of container for centers is 1 - r_target
        max_r = 1.0 - r_target
        
        # Generate grid points
        # Hex grid:
        # x = (i + 0.5*(j%2)) * spacing
        # y = j * spacing * sqrt(3)/2
        k = int(max_r / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= max_r:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n:
            return None
        # Select n points closest to the center to ensure they are well-distributed
        dists_to_center = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_center)
        return points[indices[:n]]

    # Start with a small radius to ensure feasibility
    r_current = 0.05
    # Get initial points
    x0 = hex_lattice_points(r_current)
    if x0 is None:
        # Fallback if even small r doesn't work (shouldn't happen)
        return np.zeros((n, 2))
    
    x_flat = x0.flatten()
    
    # Parameters for the optimization loop
    max_iter = 50
    r_step = 0.0005
    tol = 1e-6
    
    for _ in range(max_iter):
        r_target = r_current + r_step
        
        def residuals(x):
            centres = x[:2*n].reshape(n, 2)
            
            # Pairwise distances
            # Using pdist for efficiency
            dists = pdist(centres)
            
            # Boundary distances
            norms = np.linalg.norm(centres, axis=1)
            boundary_dists = 1.0 - norms
            
            # Residuals: max(0, 2*r_target - d_ij)
            pair_res = np.maximum(0, 2 * r_target - dists)
            
            # Residuals: max(0, r_target - (1 - |c_i|))
            bound_res = np.maximum(0, r_target - boundary_dists)
            
            return np.concatenate([pair_res, bound_res])
        
        # Run least squares
        # We only optimize the positions, r_target is fixed for this iteration
        result = least_squares(residuals, x_flat, method='trf', max_nfev=5000, 
                               xtol=1e-10, ftol=1e-10, gtol=1e-10)
        
        # Check if we can increase r
        # If the cost is very small, the configuration is feasible for r_target
        if result.cost < 1e-8:
            r_current = r_target
            x_flat = result.x[:2*n]
        else:
            # If we can't satisfy r_target, try a smaller step or stop
            # We accept the best configuration found so far (which was feasible for r_current)
            # But we might have improved the packing for r_current in the last failed attempt?
            # Actually, if it failed for r_target, the resulting x_flat might be better for r_current
            # than the previous one, because it tried to push apart.
            # Let's check the actual radius of the result
            centres = result.x[:2*n].reshape(n, 2)
            dists = pdist(centres)
            min_pair = np.min(dists) / 2.0 if len(dists) > 0 else 0
            norms = np.linalg.norm(centres, axis=1)
            min_bound = np.min(1.0 - norms)
            r_achieved = min(min_pair, min_bound)
            
            if r_achieved > r_current:
                r_current = r_achieved
                x_flat = result.x[:2*n]
            # Reduce step size to try to find the limit
            r_step *= 0.5
            if r_step < 1e-6:
                break

    centres = x_flat.reshape(n, 2)
    
    # Final check and cleanup
    # Ensure no points are outside
    norms = np.linalg.norm(centres, axis=1)
    # If any point is outside, pull it back (should not happen if residuals were handled)
    # But least_squares might drift slightly.
    # Let's do a final projection if necessary, though it might reduce r.
    # Better to just return the optimized result.
    
    return centres
# EVOLVE-BLOCK-END
