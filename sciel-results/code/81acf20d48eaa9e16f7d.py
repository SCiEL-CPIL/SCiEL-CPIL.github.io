"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Initial configuration: hexagonal-like lattice
    # For 200 circles, use a hexagonal arrangement
    # Hexagonal packing: rows with alternating offsets
    # Side length 1, so radius ~ 1/(2*sqrt(N)) * sqrt(2/sqrt(3)) for hex
    # Let's create a hexagonal grid
    
    # Number of rows and columns for hexagonal arrangement
    # In hexagonal packing, each row has either n or n-1 circles
    # Let's aim for about 14-15 rows
    
    best_config = None
    best_score = -1
    
    # Try multiple initial configurations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - t_start > deadline - 5:
            break
        
        # Hexagonal initial configuration
        # For 200 circles, use rows of varying length
        # Hexagonal: row i has circles at x positions with offset for odd rows
        # Vertical spacing: r*sqrt(3), horizontal spacing: 2r
        
        # Let's use a parameterized hexagonal grid
        # Number of rows
        n_rows = 14
        # Distribute 200 circles among 14 rows
        base = 200 // n_rows  # 14
        rem = 200 % n_rows     # 4
        # Rows 0-3 get 15 circles, rows 4-13 get 14 circles
        row_counts = [base + (1 if i < rem else 0) for i in range(n_rows)]
        
        # For hexagonal packing with varying row counts:
        # Even rows (0, 2, 4, ...) have circles at positions
        # Odd rows are offset by half the horizontal spacing
        
        # Let's use a simpler approach: create a hexagonal lattice and clip
        # Hexagonal lattice parameters
        # Horizontal spacing: dx, vertical spacing: dy = dx * sqrt(3)/2
        
        # For 200 circles in unit square:
        # If we have n_cols columns and n_rows rows in hex arrangement:
        # Total circles ≈ n_cols * n_rows (with some variation for hex)
        # Width: (n_cols-1)*dx + 2r <= 1
        # Height: (n_rows-1)*dy + 2r <= 1
        
        # Let's parameterize by dx (horizontal spacing = 2r in close packing)
        # Actually, in hexagonal close packing:
        # Horizontal distance between adjacent circles in same row: 2r
        # Vertical distance between rows: r*sqrt(3)
        # Odd rows offset by r horizontally
        
        # For a hexagonal grid with n_cols per row and n_rows rows:
        # Width needed: (n_cols - 1) * 2r + 2r = n_cols * 2r (for even rows)
        # But odd rows are shifted, so they need: (n_cols - 1) * 2r + 2r + r = ... 
        # Actually let me just place them and optimize
        
        # Simple hexagonal initial guess
        # Assume roughly 14 rows, ~14-15 per row
        n_cols = 14  # approximate
        
        # Generate hexagonal positions
        points = []
        # Vertical spacing
        dy = 1.0 / (n_rows - 1 + 1)  # rough
        # Let's compute properly
        # If n_rows rows, vertical extent = (n_rows-1) * dy + 2r
        # We want this <= 1, so dy = (1 - 2r) / (n_rows - 1)
        # For hexagonal, dy = r * sqrt(3)
        # So r = (1 - 2r) / ((n_rows-1) * sqrt(3))
        # r * (1 + 2/(n_rows-1)/sqrt(3)) = 1/sqrt(3)
        # This is getting complicated. Let me just use a parameter and optimize.
        
        # Simpler: use a regular hexagonal lattice with adjustable scale
        # Start with a reasonable r
        r_init = 0.035  # slightly below optimal
        
        # Create hexagonal lattice
        dx = 2 * r_init  # horizontal spacing
        dy = r_init * np.sqrt(3)  # vertical spacing
        
        # Number of rows to fit in height 1:
        # (n_rows - 1) * dy + 2*r_init <= 1
        # n_rows <= (1 - 2*r_init) / dy + 1
        max_rows = int((1 - 2*r_init) / dy + 1)
        # Number of cols to fit in width 1:
        # (n_cols - 1) * dx + 2*r_init <= 1 for even rows
        # n_cols <= (1 - 2*r_init) / dx + 1
        max_cols = int((1 - 2*r_init) / dx + 1)
        
        # Generate hexagonal points
        hex_points = []
        for row in range(max_rows + 1):
            y = -0.5 + r_init + row * dy
            if y > 0.5 - r_init + 1e-9:
                break
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(max_cols + 1):
                x = -0.5 + r_init + offset + col * dx
                if x > 0.5 - r_init + 1e-9:
                    break
                hex_points.append([x, y])
        
        hex_points = np.array(hex_points)
        
        if len(hex_points) < N:
            # Not enough points, adjust
            continue
        
        # Take first N points (or randomly sample)
        if len(hex_points) >= N:
            if restart == 0:
                # Take a contiguous block
                idx = np.arange(N)
            else:
                idx = rng.choice(len(hex_points), size=N, replace=False)
            config = hex_points[idx].copy()
        else:
            config = hex_points.copy()
            # Pad with random points
            while len(config) < N:
                config = np.vstack([config, rng.uniform(-0.4, 0.4, size=(1, 2))])
            config = config[:N]
        
        # Now optimize: maximize the minimum clearance
        # Use a force-directed approach
        # Objective: maximize min(min_pair_dist/2, min_boundary_dist)
        # Equivalent to: for a target radius r, check if all constraints are satisfied
        # Use iterative improvement
        
        # Force-directed optimization
        # For each step, compute forces:
        # - Pairwise: if dist < 2r, push apart
        # - Boundary: if dist to wall < r, push inward
        
        # We'll use a simple iterative method
        config_opt = config.copy()
        
        # Estimate initial radius
        # Compute current min clearance
        def compute_radius(pts):
            # Pairwise distances
            n = len(pts)
            min_pair = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_pair:
                        min_pair = d
            # Boundary distances
            min_bound = np.inf
            for i in range(n):
                d_left = pts[i, 0] + 0.5
                d_right = 0.5 - pts[i, 0]
                d_bottom = pts[i, 1] + 0.5
                d_top = 0.5 - pts[i, 1]
                min_bound = min(min_bound, d_left, d_right, d_bottom, d_top)
            return min(min_pair / 2, min_bound)
        
        # Use a smarter optimization: gradient ascent on the min clearance
        # Or use scipy minimize on a reformulated problem
        
        # Let's use a penalty-based approach
        # Minimize: -min_clearance + penalty for violations
        # But scipy minimize might be slow for 400 variables
        
        # Instead, use a simple iterative force-directed method
        n_iter = 500
        lr = 0.01
        
        for it in range(n_iter):
            if time.time() - t_start > deadline - 10:
                break
            
            # Compute pairwise distances
            # This is O(N^2) which is 200^2 = 40000, fast enough
            diffs = config_opt[:, np.newaxis, :] - config_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            
            # Find current minimum pair distance
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Compute boundary distances
            bound_dists = np.min([
                config_opt[:, 0] + 0.5,  # left
                0.5 - config_opt[:, 0],  # right
                config_opt[:, 1] + 0.5,  # bottom
                0.5 - config_opt[:, 1]   # top
            ], axis=0)
            min_bound_dist = np.min(bound_dists)
            
            # Current radius
            r_curr = min(min_pair_dist / 2, min_bound_dist)
            
            # Compute forces
            forces = np.zeros_like(config_opt)
            
            # Pairwise forces: push apart if too close
            # Use a soft potential
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2 * r_curr * 1.1:  # within influence
                        # Force magnitude
                        f_mag = 1.0 / (d + 1e-10)
                        diff = config_opt[i] - config_opt[j]
                        forces[i] += f_mag * diff
                        forces[j] -= f_mag * diff
            
            # Boundary forces: push inward if too close
            for i in range(N):
                # Left wall
                d_left = config_opt[i, 0] + 0.5
                if d_left < r_curr * 1.1:
                    forces[i, 0] += 1.0 / (d_left + 1e-10)
                # Right wall
                d_right = 0.5 - config_opt[i, 0]
                if d_right < r_curr * 1.1:
                    forces[i, 0] -= 1.0 / (d_right + 1e-10)
                # Bottom wall
                d_bottom = config_opt[i, 1] + 0.5
                if d_bottom < r_curr * 1.1:
                    forces[i, 1] += 1.0 / (d_bottom + 1e-10)
                # Top wall
                d_top = 0.5 - config_opt[i, 1]
                if d_top < r_curr * 1.1:
                    forces[i, 1] -= 1.0 / (d_top + 1e-10)
            
            # Update
            config_opt += lr * forces
            
            # Clip to valid region (with small margin)
            margin = 1e-6
            config_opt[:, 0] = np.clip(config_opt[:, 0], -0.5 + margin, 0.5 - margin)
            config_opt[:, 1] = np.clip(config_opt[:, 1], -0.5 + margin, 0.5 - margin)
        
        # Compute final score
        final_radius = compute_radius(config_opt)
        
        if final_radius > best_score:
            best_score = final_radius
            best_config = config_opt.copy()
        
        # Try a few more restarts if time permits
        if time.time() - t_start > deadline - 20:
            break
    
    # Final validation
    if best_config is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N]
    
    # Ensure valid
    best_config = np.clip(best_config, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_config

# EVOLVE-BLOCK-END
