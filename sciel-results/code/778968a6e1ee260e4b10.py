import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    # 1. Initialization: Hexagonal Grid
    # Create a hexagonal lattice
    # Side of hexagon (distance between centers in hex packing)
    # We want to fit as many as possible, then pick 200.
    # Or just create a grid slightly denser than needed.
    
    # Let's generate points in a hex pattern
    # Spacing s
    # Let's try a few spacings or just generate a large set and pick 200
    # Better: Generate a hex grid that covers the square, then filter to 200.
    
    # Generate hex grid
    s = 0.04  # Initial spacing, roughly 1/sqrt(200) ~ 0.07, but hex is denser. 0.04 gives ~2500 points
    # Actually, let's just create a grid of ~400 points and pick the best 200?
    # Or create a hex grid with spacing 1/sqrt(N) * factor.
    
    # Let's use a simple hex grid generation
    def generate_hex_grid(spacing, bounds):
        points = []
        h = spacing * np.sqrt(3) / 2
        x = bounds[0][0]
        while x <= bounds[0][1]:
            y = bounds[1][0]
            offset = 0 if int(x / spacing) % 2 == 0 else h / 2 # This logic is tricky for centered grid
            # Let's do it more robustly
            break
        # Simpler: just use a square grid with alternating row offsets
        # Rows
        n_rows = int((bounds[1][1] - bounds[1][0]) / h) + 1
        n_cols = int((bounds[0][1] - bounds[0][0]) / spacing) + 1
        
        pts = []
        for i in range(n_rows):
            for j in range(n_cols):
                x = bounds[0][0] + j * spacing + (0 if i % 2 == 0 else spacing / 2)
                y = bounds[1][0] + i * h
                # Center the grid
                # We will center it later or generate centered
                pts.append((x, y))
        return np.array(pts)

    # Let's generate a grid centered at 0,0
    # Target area [-0.5, 0.5]
    # Let's generate a larger grid and crop
    s = 0.03 
    h = s * np.sqrt(3) / 2
    x_range = np.arange(-0.6, 0.6, s)
    y_range = np.arange(-0.6, 0.6, h)
    
    hex_pts = []
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            # Offset every other row
            x_off = x + (s / 2 if i % 2 == 0 else 0)
            hex_pts.append((x_off, y))
            
    hex_pts = np.array(hex_pts)
    
    # Filter points that are strictly inside the square (with margin)
    # Actually, we want points inside [-0.5, 0.5]
    mask = (hex_pts[:, 0] > -0.5 + s/2) & (hex_pts[:, 0] < 0.5 - s/2) & (hex_pts[:, 1] > -0.5 + s/2) & (hex_pts[:, 1] < 0.5 - s/2)
    hex_pts = hex_pts[mask]
    
    # If we have more than N, pick the N closest to the center?
    # Or random sample?
    # Picking closest to center tends to fill the middle, leaving corners empty.
    # For packing, we want to fill the square.
    # Let's just take a random subset of 200 from the valid hex points.
    # Or better, take all valid points, and if > 200, remove the ones with lowest "value" (e.g. closest to boundary? No, we want to keep boundary ones to maximize radius).
    # Actually, a good init is a grid that is slightly compressed.
    
    # Let's just use the first 200 points from a sorted list by distance to center?
    # No, that clusters in the middle.
    # Let's use a random selection to break symmetry, or just take a grid of 15x14=210 points.
    
    # Let's try a rectangular grid of 15x14 = 210 points, then remove 10.
    # Or just use the hex points.
    if len(hex_pts) > N:
        # Select N points. To ensure good spread, we can use farthest-point sampling or just random.
        # Random is fine for local search.
        indices = np.random.choice(len(hex_pts), N, replace=False)
        x0 = hex_pts[indices]
    else:
        # If not enough, pad with random points inside
        x0 = hex_pts.copy()
        while len(x0) < N:
            x0 = np.vstack([x0, np.random.uniform(-0.4, 0.4, (1, 2))])
    
    # 2. Optimization
    # Objective: Maximize soft-min of distances
    # d_ij = ||x_i - x_j||
    # d_ik = min(0.5 - |x_i|, 0.5 - |y_i|)
    
    def objective(x_flat, beta):
        x = x_flat.reshape(N, 2)
        
        # Pairwise distances
        # Use broadcasting
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq[mask])
        
        # Wall distances
        # Distance to each wall
        d_left = 0.5 - x[:, 0]
        d_right = x[:, 0] + 0.5
        d_bottom = 0.5 - x[:, 1]
        d_top = x[:, 1] + 0.5
        
        # Min distance to any wall for each point
        # d_wall_i = min(d_left, d_right, d_bottom, d_top)
        d_walls = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
        
        # Soft min
        # We want to maximize the min of all dists and d_walls.
        # Combined list of all critical distances
        all_dists = np.concatenate([dists, d_walls])
        
        # Handle zeros or negatives (should not happen if init is valid, but just in case)
        # Clip to small positive
        all_dists = np.clip(all_dists, 1e-6, None)
        
        # Log-sum-exp for soft min: -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow, use max trick
        max_d = np.max(all_dists)
        sum_exp = np.sum(np.exp(-beta * (all_dists - max_d)))
        soft_min = max_d - (1.0 / beta) * np.log(sum_exp)
        
        return -soft_min  # Minimize negative soft_min

    def gradient(x_flat, beta):
        # Numerical gradient or analytical?
        # Analytical is faster.
        # d/dx_i ( -1/beta log(sum_j exp(-beta d_j)) )
        # = -1/beta * (1/sum) * sum_j (-exp(-beta d_j) * (-beta * d/dx_i d_j))
        # = -1/sum * sum_j exp(-beta d_j) * d/dx_i d_j
        # = - sum_j ( w_j * d/dx_i d_j )  where w_j = exp(-beta d_j) / sum(exp(-beta d_k))
        
        x = x_flat.reshape(N, 2)
        
        # Pairwise
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq[mask])
        
        # Wall distances
        d_left = 0.5 - x[:, 0]
        d_right = x[:, 0] + 0.5
        d_bottom = 0.5 - x[:, 1]
        d_top = x[:, 1] + 0.5
        d_walls = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
        
        all_dists = np.concatenate([dists, d_walls])
        all_dists = np.clip(all_dists, 1e-6, None)
        
        max_d = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - max_d))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        grad = np.zeros_like(x)
        
        # Pairwise gradient contribution
        # d/dx_i d_ij = (x_i - x_j) / d_ij
        # We need to map weights back to pairs
        # This is tricky with flat arrays.
        # Let's just use numerical gradient if analytical is too complex to implement quickly and correctly.
        # N=200, 400 vars. NumPy finite diff is slow.
        # Let's implement analytical carefully.
        
        # We have weights for all pairs (N*(N-1)/2) and walls (N*4? No, N mins).
        # The min for walls is non-smooth. 
        # For simplicity and robustness, let's use `scipy.optimize.minimize` with numerical gradient 
        # but limit iterations? 
        # Or use a simpler heuristic: "Nudge" method.
        
        # Given the time constraint and complexity, let's switch to a simpler, robust local search:
        # 1. Calculate min radius r.
        # 2. Find all pairs with dist < 2*r + eps and walls with dist < r + eps.
        # 3. Move points apart.
        
        return None # Placeholder

    # Let's use a simpler optimization strategy: 
    # Iterative local search.
    # 1. Calculate current min radius.
    # 2. Identify "active" constraints.
    # 3. Apply forces.
    
    def compute_radius(x):
        # x: (N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dists_sq))
        
        d_left = 0.5 - x[:, 0]
        d_right = x[:, 0] + 0.5
        d_bottom = 0.5 - x[:, 1]
        d_top = x[:, 1] + 0.5
        min_wall_dist = np.min(np.stack([d_left, d_right, d_bottom, d_top]))
        
        return min(min_pair_dist / 2, min_wall_dist)

    best_x = x0.copy()
    best_r = compute_radius(best_x)
    
    # Simple Gradient Ascent on a smooth proxy: Sum of inverse squared distances?
    # Or just use the soft-min with numerical gradient via `scipy` but with a tight budget.
    
    # Let's try a "Repulsive Force" simulation.
    # Force F_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^3 + Wall_Forces
    # This is like a charge repulsion.
    # We want to maximize the min distance. 
    # A common trick: maximize the sum of 1/d^2? No, that minimizes d.
    # We want to *increase* d.
    # So force should be repulsive.
    # If we simulate particles repelling each other and being pushed from walls, they will spread out.
    
    step_size = 0.005
    decay = 0.99
    
    for step in range(5000):
        if time.time() - start_time > deadline:
            break
            
        x = best_x.copy()
        
        # Calculate forces
        forces = np.zeros_like(x)
        
        # Pairwise repulsion
        # Vectorized
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0) # Avoid div by 0
        
        dists = np.sqrt(dists_sq)
        
        # Force magnitude: 1 / d^2 (Coulomb-like)
        # Direction: diff / d
        # So Force = diff / d^3
        inv_d3 = 1.0 / (dists_sq * dists)
        np.fill_diagonal(inv_d3, 0.0)
        
        pair_forces = diff * inv_d3[:, :, np.newaxis]
        forces += np.sum(pair_forces, axis=1)
        
        # Wall forces
        # Push away from walls if close
        # d_left = 0.5 - x_i0. If d_left < r, push right.
        # Let's use a soft wall: Force = 1/d^2 if d < threshold?
        # Or just project back if outside?
        # Better: Add a force term that grows as we approach the wall.
        
        d_left = 0.5 - x[:, 0]
        d_right = x[:, 0] + 0.5
        d_bottom = 0.5 - x[:, 1]
        d_top = x[:, 1] + 0.5
        
        # Threshold
        thresh = 0.05
        
        # Force x
        f_x = np.zeros(N)
        f_y = np.zeros(N)
        
        # Left wall: if d_left < thresh, force is positive (right)
        mask_l = d_left < thresh
        f_x[mask_l] += (1.0 / (d_left[mask_l] + 1e-6)**2)
        
        # Right wall: if d_right < thresh, force is negative (left)
        mask_r = d_right < thresh
        f_x[mask_r] -= (1.0 / (d_right[mask_r] + 1e-6)**2)
        
        # Bottom wall: if d_bottom < thresh, force is positive (up)
        mask_b = d_bottom < thresh
        f_y[mask_b] += (1.0 / (d_bottom[mask_b] + 1e-6)**2)
        
        # Top wall: if d_top < thresh, force is negative (down)
        mask_t = d_top < thresh
        f_y[mask_t] -= (1.0 / (d_top[mask_t] + 1e-6)**2)
        
        forces[:, 0] += f_x
        forces[:, 1] += f_y
        
        # Update positions
        # Normalize forces?
        # max_force = np.max(np.linalg.norm(forces, axis=1))
        # if max_force > 0:
        #     forces = forces / max_force * step_size
        # else:
        #     break
        
        x += forces * step_size
        
        # Clamp to square
        x[:, 0] = np.clip(x[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
        x[:, 1] = np.clip(x[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
        
        r = compute_radius(x)
        
        if r > best_r:
            best_r = r
            best_x = x.copy()
            
        step_size *= decay
        
        # If step size is too small, break
        if step_size < 1e-6:
            break
            
    # Final check
    r = compute_radius(best_x)
    
    return best_x
