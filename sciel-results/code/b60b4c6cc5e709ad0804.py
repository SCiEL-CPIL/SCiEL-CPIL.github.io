"""Program for packing 50 equal spheres in a unit cube; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC-inspired initialization followed by local optimization to maximize
    the minimum distance between centers and to the cube boundaries.
    """
    np.random.seed(42)
    
    # Initialize with a BCC-like lattice pattern
    # BCC has points at corners and body center of each sub-cube
    # We'll create a grid and add body-center points, then select 50
    
    # Start with a simple approach: create a denser grid and use BCC offset
    m = 4  # 4x4x4 grid gives 64 points, we need 50
    spacing = 1.0 / m
    centers = []
    
    # Add corner-like points (shifted inward) and body-center points
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner point of the sub-cell
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                
                # Body center point (offset by half spacing in all directions)
                # Only add if within bounds
                bx = x + spacing / 2
                by = y + spacing / 2
                bz = z + spacing / 2
                if abs(bx) <= 0.5 and abs(by) <= 0.5 and abs(bz) <= 0.5:
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    
    # If we have more than N points, select N points that are most spread out
    if len(centers) > N:
        # Use a greedy farthest-point sampling
        selected = [np.random.choice(len(centers))]
        while len(selected) < N:
            # For each unselected point, compute min distance to selected
            min_dists = np.inf
            for idx in range(len(centers)):
                if idx in selected:
                    continue
                d = np.min(np.linalg.norm(centers[idx] - centers[selected], axis=1))
                min_dists = min(min_dists, d) if False else min_dists  # This is inefficient, let's do it properly
            
            # Better approach: compute for all unselected at once
            unselected = [i for i in range(len(centers)) if i not in selected]
            best_idx = -1
            best_min_dist = -1
            for idx in unselected:
                dists = np.linalg.norm(centers[idx] - centers[selected], axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        
        centers = centers[selected]
    elif len(centers) < N:
        # Pad with random points if needed (shouldn't happen with m=4)
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.4, 0.4, 3)])
    
    centers = centers[:N]
    
    # Define the objective function: maximize minimum distance
    # We'll minimize the negative of the minimum distance
    
    def objective(x):
        """Return negative of the minimum sphere radius (to be minimized)."""
        pts = x.reshape(N, 3)
        
        # Check bounds: all points must be within [-0.5, 0.5]
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10  # Large penalty
        
        # Compute all pairwise distances
        from scipy.spatial.distance import cdist
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Distance to boundaries
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        
        # The radius is half of the minimum of pair distances and boundary distances
        r = 0.5 * min(min_pair_dist, min_bound_dist)
        
        return -r
    
    # Use Powell for local optimization (no gradient needed, robust for non-smooth)
    # Start from the BCC-inspired initialization
    x0 = centers.flatten()
    
    # Run optimization with multiple restarts to avoid local minima
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try more restarts with slight perturbations
    for restart in range(10):
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + np.random.normal(0, 0.01, N * 3)
            # Clamp to valid range
            x_init = np.clip(x_init.reshape(N, 3), -0.49, 0.49).flatten()
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Powell',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Final result
    result_centers = best_x.reshape(N, 3)
    
    # Ensure all points are strictly inside the cube
    result_centers = np.clip(result_centers, -0.4999, 0.4999)
    
    return result_centers
# EVOLVE-BLOCK-END
