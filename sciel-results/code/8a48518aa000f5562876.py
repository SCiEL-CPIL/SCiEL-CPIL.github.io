"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize


def _radius(pts):
    """Compute the packing radius: min of half-min-pair-distance and min-distance-to-boundary."""
    if len(pts) < 2:
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(pts), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    tree = cKDTree(pts)
    # Query nearest neighbor for each point
    dists, _ = tree.query(pts, k=2)
    min_pair = np.min(dists[:, 1])
    return min(min_boundary, min_pair / 2.0)


def _make_hex_grid(n):
    """Create a hexagonal packing grid of n points, scaled to fit in [-0.5, 0.5]^2."""
    # Start with a hexagonal lattice
    # Hexagonal spacing: dx = 1, dy = sqrt(3)/2
    # We'll generate a large enough grid and then select n points
    # Center at origin, scale to fit
    
    # Generate hex lattice points
    # Use a rectangular region with hex offset
    cols = 20
    rows = 15
    pts = []
    for j in range(rows):
        for i in range(cols):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    
    # Center
    pts -= pts.mean(axis=0)
    
    # Scale to fit in [-0.5, 0.5]^2
    max_extent = np.max(np.abs(pts))
    if max_extent > 0:
        pts /= max_extent * 2.0  # Scale so max |coord| = 0.5
    
    # Select n points closest to center for better packing
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)[:n]
    pts = pts[indices]
    
    return pts


def _random_perturbation(pts, sigma=0.01):
    """Add small random perturbation."""
    return pts + np.random.normal(0, sigma, size=pts.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with hexagonal grid
    pts = _make_hex_grid(N)
    best_pts = pts.copy()
    best_radius = _radius(pts)
    
    # Nelder-Mead optimization to maximize radius
    # We minimize the negative radius
    def objective(x):
        return -_radius(x.reshape(-1, 2))
    
    x0 = pts.flatten()
    
    # Run Nelder-Mead with multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0
        else:
            x_start = _random_perturbation(best_pts, sigma=0.02).flatten()
        
        try:
            result = minimize(
                objective,
                x_start,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'maxiter': 5000,
                    'maxfev': 20000
                }
            )
            
            current_radius = -result.fun
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = result.x.reshape(-1, 2)
        except Exception:
            continue
        
        if time.time() > deadline - 30:
            break
    
    # Final validation and clipping to ensure points are inside the square
    # Ensure all points are strictly inside [-0.5, 0.5]^2
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    # Verify the result is valid
    final_radius = _radius(best_pts)
    if final_radius <= 0:
        # Fallback to initial grid
        return _make_hex_grid(N)
    
    return best_pts

# EVOLVE-BLOCK-END
