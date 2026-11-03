"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_centers = None
    best_radius = 0.0
    
    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        # Check bounds
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        
        # Compute pairwise distances
        n = len(centers)
        min_pair_dist = np.inf
        for i in range(n):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                min_pair_dist = min(min_pair_dist, np.min(dists))
        
        if not np.isfinite(min_pair_dist):
            return 0.0
            
        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)
    
    def make_hexagonal_layout(offset_x, offset_y):
        """Create a hexagonal packing layout."""
        centers = []
        # Hexagonal lattice parameters
        # For hexagonal packing, rows are offset
        # Spacing between centers: d
        # We want to fit as many as possible in the square
        
        # Try different hexagonal configurations
        # Hexagonal lattice: 
        #   row spacing = d * sqrt(3)/2
        #   col spacing = d
        #   odd rows offset by d/2
        
        # We'll place circles in a hex pattern and then adjust
        # Start with a reasonable estimate of the spacing
        # For N circles in unit square, rough area estimate:
        # N * (pi * r^2) <= 1, so r <= 1/sqrt(N*pi) ~ 0.04
        # But packing efficiency ~ 0.9069 for hex, so r ~ 0.0366
        
        # Let's try a grid of hexagonal positions
        # Number of rows and columns to try
        for n_rows in range(10, 20):
            if time.time() > deadline - 10:
                break
            n_cols = math.ceil(N / n_rows)
            
            # Hexagonal spacing
            # Width needed: (n_cols - 1) * d + d = n_cols * d (with offset)
            # Actually: even rows have n_cols points, odd rows have n_cols-1 or n_cols
            # Let's use: d = 1.0 / (n_cols + 0.5) approximately
            
            # Try multiple spacings
            for d_scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
                d = d_scale / (n_cols + 0.5)
                row_spacing = d * math.sqrt(3) / 2
                
                # Check if this fits in height
                total_height = (n_rows - 1) * row_spacing + d
                if total_height > 1.0 + 1e-6:
                    continue
                
                # Generate hexagonal points
                pts = []
                for row in range(n_rows):
                    y = -0.5 + d/2 + row * row_spacing
                    if row % 2 == 0:
                        n_in_row = n_cols
                        x_start = -0.5 + d/2
                    else:
                        n_in_row = n_cols - 1
                        x_start = -0.5 + d
                    
                    for col in range(n_in_row):
                        x = x_start + col * d
                        pts.append([x, y])
                
                pts = np.array(pts)
                if len(pts) >= N:
                    pts = pts[:N]
                elif len(pts) > 0:
                    # Pad with random points near existing ones if needed
                    pass
                
                if len(pts) == N:
                    r = compute_radius(pts)
                    if r > best_radius:
                        best_radius = r
                        best_centers = pts.copy()
        
        # Also try a simple square grid with random perturbation
        m = int(np.ceil(np.sqrt(N)))
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            grid = grid[:N]
            
            # Add small random perturbation
            if trial > 0:
                noise = np.random.randn(N, 2) * 0.005
                grid = grid + noise
            
            r = compute_radius(grid)
            if r > best_radius:
                best_radius = r
                best_centers = grid.copy()
        
        # Refine the best configuration found so far
        if best_centers is not None:
            centers = best_centers.copy()
            
            # Coordinate ascent refinement
            max_iter = 50
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    
                    orig = centers[i].copy()
                    best_pos = orig
                    best_r = compute_radius(centers)
                    
                    # Try moving in small steps
                    for dx in [-0.002, -0.001, 0, 0.001, 0.002]:
                        for dy in [-0.002, -0.001, 0, 0.001, 0.002]:
                            if dx == 0 and dy == 0:
                                continue
                            new_center = orig + np.array([dx, dy])
                            # Check bounds
                            if new_center[0] < -0.5 or new_center[0] > 0.5:
                                continue
                            if new_center[1] < -0.5 or new_center[1] > 0.5:
                                continue
                            
                            centers[i] = new_center
                            r = compute_radius(centers)
                            if r > best_r:
                                best_r = r
                                best_pos = new_center.copy()
                    
                    if not np.allclose(best_pos, orig):
                        centers[i] = best_pos
                        improved = True
                
                if not improved:
                    break
            
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Try scipy-based optimization on the best configuration
        if best_centers is not None and time.time() < deadline - 20:
            centers = best_centers.copy()
            
            def objective(x):
                c = x.reshape(-1, 2)
                # Penalize going out of bounds
                bound_penalty = 0.0
                for i in range(N):
                    if c[i, 0] < -0.5 or c[i, 0] > 0.5 or c[i, 1] < -0.5 or c[i, 1] > 0.5:
                        bound_penalty += 100.0 * (abs(c[i, 0]) - 0.5 + abs(c[i, 1]) - 0.5)
                
                # Compute min pairwise distance
                min_pd = np.inf
                for i in range(N):
                    diffs = c[i+1:] - c[i]
                    dists = np.sqrt(np.sum(diffs**2, axis=1))
                    if len(dists) > 0:
                        min_pd = min(min_pd, np.min(dists))
                
                min_wall = np.min(0.5 - np.abs(c))
                r = min(min_pd / 2.0, min_wall)
                return -r + bound_penalty
            
            # Use L-BFGS-B with bounds
            bounds = [(-0.5, 0.5)] * (2 * N)
            x0 = centers.flatten()
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.success or result.fun < -best_radius:
                    new_centers = result.x.reshape(-1, 2)
                    r = compute_radius(new_centers)
                    if r > best_radius:
                        best_radius = r
                        best_centers = new_centers.copy()
            except Exception:
                pass
        
        # Final polish with very small steps
        if best_centers is not None and time.time() < deadline - 3:
            centers = best_centers.copy()
            for step in [0.0005, 0.0002, 0.0001]:
                for i in range(N):
                    if time.time() > deadline - 3:
                        break
                    orig = centers[i].copy()
                    best_pos = orig
                    best_r = compute_radius(centers)
                    
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            new_center = orig + np.array([dx, dy])
                            if new_center[0] < -0.5 or new_center[0] > 0.5:
                                continue
                            if new_center[1] < -0.5 or new_center[1] > 0.5:
                                continue
                            centers[i] = new_center
                            r = compute_radius(centers)
                            if r > best_r:
                                best_r = r
                                best_pos = new_center.copy()
                    
                    centers[i] = best_pos
            
            r = compute_radius(centers)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
