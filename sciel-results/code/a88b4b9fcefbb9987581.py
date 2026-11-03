"""Program for packing 50 equal spheres in a unit cube; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def construct():
    """
    Construct centers for 50 equal spheres in the unit cube [-0.5, 0.5]^3.
    Uses a BCC lattice initialization followed by iterative repulsion-based relaxation
    and local optimization to maximize the minimum distance between centers and to
    the cube boundaries.
    """
    np.random.seed(42)
    
    # Initialize with a proper BCC lattice scaled to fit the unit cube
    # BCC: simple cubic lattice + body-centered points
    # For a cube of side 1, BCC has lattice constant a = 1/n per unit cell
    # We want approximately 50 points total
    
    # BCC in a box of side L: n x n x n unit cells gives 2*n^3 points
    # For n=3: 54 points, close to 50
    # For n=2: 16 points, too few
    # Let's use n=3 and select the best 50
    
    def generate_bcc(n, L=1.0):
        """Generate BCC lattice points in a cube of side L centered at origin."""
        a = L / n  # lattice constant
        points = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Corner point
                    x = -L/2 + (i + 0.5) * a
                    y = -L/2 + (j + 0.5) * a
                    z = -L/2 + (k + 0.5) * a
                    points.append([x, y, z])
                    # Body center point
                    bx = x + a/2
                    by = y + a/2
                    bz = z + a/2
                    # Only add if within bounds
                    if abs(bx) <= L/2 and abs(by) <= L/2 and abs(bz) <= L/2:
                        points.append([bx, by, bz])
        return np.array(points)
    
    # Generate BCC with n=3 (gives 54 points)
    bcc_points = generate_bcc(3)
    
    # Select 50 points using farthest-point sampling for better spread
    if len(bcc_points) >= N:
        # Farthest point sampling
        selected = [0]  # Start with first point
        while len(selected) < N:
            unselected = [i for i in range(len(bcc_points)) if i not in selected]
            best_idx = -1
            best_min_dist = -1
            for idx in unselected:
                dists = np.linalg.norm(bcc_points[idx] - bcc_points[selected], axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(best_idx)
        centers = bcc_points[selected].copy()
    else:
        # Pad if needed
        centers = bcc_points.copy()
        while len(centers) < N:
            centers = np.vstack([centers, np.random.uniform(-0.4, 0.4, 3)])
        centers = centers[:N]
    
    # Define the objective function: maximize minimum radius
    def objective(x):
        """Return negative of the minimum sphere radius (to be minimized)."""
        pts = x.reshape(N, 3)
        
        # Check bounds
        if np.any(np.abs(pts) > 0.5 + 1e-9):
            return 1e10
        
        # Compute all pairwise distances
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Distance to boundaries
        dist_to_bound = 0.5 - np.max(np.abs(pts), axis=1)
        min_bound_dist = np.min(dist_to_bound)
        
        # The radius is half of the minimum of pair distances and boundary distances
        r = 0.5 * min(min_pair_dist, min_bound_dist)
        
        return -r
    
    # First, do a repulsion-based relaxation to get a better starting point
    # This helps escape poor local configurations
    def repulsion_relax(pts, n_iter=200, step_size=0.01):
        """Iteratively push points apart based on inverse-square repulsion."""
        pts = pts.copy()
        for it in range(n_iter):
            # Compute forces
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        # Random small perturbation
                        diff = np.random.uniform(-0.01, 0.01, 3)
                        dist = np.linalg.norm(diff)
                    # Repulsive force ~ 1/dist^2
                    force_mag = step_size / (dist * dist + 1e-6)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            
            # Add boundary repulsion
            for i in range(N):
                for k in range(3):
                    if pts[i, k] > 0.4:
                        forces[i, k] -= step_size * (pts[i, k] - 0.4) * 10
                    elif pts[i, k] < -0.4:
                        forces[i, k] -= step_size * (pts[i, k] + 0.4) * 10
            
            # Update positions
            pts += forces
            pts = np.clip(pts, -0.499, 0.499)
        
        return pts
    
    # Apply repulsion relaxation
    centers = repulsion_relax(centers, n_iter=100, step_size=0.005)
    
    # Now do local optimization with Powell
    x0 = centers.flatten()
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Time budget: 110 seconds
    start_time = time.time()
    max_time = 110.0
    
    # Multiple restarts with perturbations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Larger perturbations for later restarts
            pert_scale = 0.02 * (restart + 1)
            x_init = x0 + np.random.normal(0, pert_scale, N * 3)
            x_init = np.clip(x_init.reshape(N, 3), -0.49, 0.49).flatten()
        
        try:
            result = minimize(
                objective,
                x_init,
                method='Powell',
                options={
                    'maxiter': 500,
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
