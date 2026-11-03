"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit
    
    # Function to compute the minimum radius
    def min_radius(c):
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # --- Phase 0: Generate diverse initial configurations ---
    
    # 1. Simple Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N].copy()
    
    # 2. Body-Centered Cubic (BCC) Lattice
    # BCC has 2 points per unit cell.
    # To fit 50 points, we need approx 25 cells.
    # Let's generate a 3x3x3 grid of BCC cells (27 cells * 2 = 54 points) and take 50.
    # Or a 2x2x4 grid (16 cells * 2 = 32) - too few.
    # 3x3x3 gives 54 points.
    # Scale the BCC lattice to fit in [-0.5, 0.5].
    # Standard BCC points in a unit cell [0,1]^3: (0,0,0) and (0.5,0.5,0.5).
    # We will generate a grid of these, then shift/scale to center in the cube.
    
    def generate_bcc(n_points):
        # Create a 3D grid of BCC unit cells
        # We want roughly cube_root(n_points/2) cells per side
        m_cells = int(np.ceil((n_points / 2.0) ** (1.0 / 3.0)))
        
        points = []
        # Generate corner points and body center points
        # Coordinates for a single cell at (i,j,k) with side length s:
        # Corner: (i*s, j*s, k*s)
        # Center: ((i+0.5)*s, (j+0.5)*s, (k+0.5)*s)
        
        # We generate points in a range [0, m_cells]
        # Then we will rescale to fit [-0.5, 0.5]
        
        raw_points = []
        for i in range(m_cells + 1):
            for j in range(m_cells + 1):
                for k in range(m_cells + 1):
                    raw_points.append([i, j, k])
                    # Only add body center if it's inside the "bulk" or we just add all
                    # Standard BCC includes centers for all cells.
                    # To avoid duplicates at boundaries when merging cells, 
                    # a common trick is to generate a grid of centers and corners separately
                    # or just generate a large set and cluster/trim.
                    # Simpler: Generate a 3x3x3 array of unit cubes.
                    # Points at (i,j,k) and (i+0.5, j+0.5, k+0.5) for i,j,k in [0..m-1]
                    # This creates a dense cloud.
        
        # Let's use a cleaner construction for BCC:
        # Layer 1 (z=0): Points at (i+0.5, j+0.5) for i,j in range(m)
        # Layer 2 (z=0.5): Points at (i, j) for i,j in range(m) -- wait, BCC alternates.
        
        # Standard BCC:
        # At z=0: (i, j, 0) for i,j in 0..m-1
        # At z=0.5: (i+0.5, j+0.5, 0.5) for i,j in 0..m-1
        # At z=1.0: (i, j, 1.0) ...
        
        # Let's generate a 3x3x3 block of BCC unit cells.
        # Total points = 2 * 3 * 3 * 3 = 54.
        pts = []
        m_bcc = 3
        for i in range(m_bcc):
            for j in range(m_bcc):
                for k in range(m_bcc):
                    # Corner of the cell (shared, but we just generate all and dedupe/trim later? No, just generate unique set)
                    # To get a proper BCC lattice, we define the basis.
                    # Basis 1: (0,0,0)
                    # Basis 2: (0.5, 0.5, 0.5)
                    # Lattice vectors: (1,0,0), (0,1,0), (0,0,1)
                    
                    # Point A: (i, j, k)
                    pts.append([i, j, k])
                    # Point B: (i+0.5, j+0.5, k+0.5)
                    pts.append([i+0.5, j+0.5, k+0.5])
        
        pts = np.array(pts)
        
        # Center the points
        center = pts.mean(axis=0)
        pts = pts - center
        
        # Scale to fit in [-0.5, 0.5] with a small margin
        # The extent of the BCC lattice is roughly m_bcc * 1.0
        # We want max abs coord to be <= 0.5 * (1 - margin)
        max_extent = np.max(np.abs(pts))
        target_extent = 0.5 * 0.95 # 5% margin
        scale = target_extent / max_extent
        pts = pts * scale
        
        # Sort by some criterion? Or just take the first N?
        # The first N might be biased towards one corner if the list is ordered.
        # Let's shuffle or select the most "central" N points?
        # For initialization, random selection from the lattice is fine.
        indices = np.random.choice(len(pts), N, replace=False)
        return pts[indices]

    init_bcc = generate_bcc(N)
    
    # 3. Random initialization
    init_rand = np.random.uniform(-0.45, 0.45, (N, 3))
    
    initializations = [init_grid, init_bcc, init_rand]
    
    best_overall_r = -1.0
    best_overall_centers = None
    
    # Run a short optimization for each initialization to find the best starting point for main search
    for init_c in initializations:
        if time.time() - start_time > time_limit * 0.1:
            break
            
        c = init_c.copy()
        r = min_radius(c)
        
        # Quick local improvement (few iterations of simple gradient ascent on radius)
        # Using a simple steepest ascent on the min distance
        for _ in range(200):
            if time.time() - start_time > time_limit * 0.1:
                break
                
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            r_curr = min(np.min(d) / 2.0, np.min(dist_to_walls))
            
            G = np.zeros_like(c)
            eps = 1e-4
            # Active constraints
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] < 2 * r_curr + eps:
                        vec = c[i] - c[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            G[i] += vec / norm
                            G[j] -= vec / norm
            
            for i in range(N):
                if dist_to_walls[i] < r_curr + eps:
                    for k in range(3):
                        if c[i, k] > 0:
                            G[i, k] -= 1.0
                        else:
                            G[i, k] += 1.0
            
            norm_G = np.linalg.norm(G)
            if norm_G > 1e-9:
                G /= norm_G
            
            step = 0.005
            c_new = c + step * G
            c_new = np.clip(c_new, -0.5 + 1e-9, 0.5 - 1e-9)
            
            r_new = min_radius(c_new)
            if r_new > r_curr:
                c = c_new
                r = r_new
            else:
                # Try smaller step or break? For init, just break if no progress for a bit?
                # Let's just continue with a smaller step
                step *= 0.5
                if step < 1e-6:
                    break
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = c.copy()

    centers = best_overall_centers
    best_r = best_overall_r
    best_centers = centers.copy()
    
    # --- Phase 1: Simulated Annealing ---
    T = 0.02
    cooling_rate = 0.99
    min_T = 1e-6
    sa_deadline = time_limit * 0.6
    
    while time.time() - start_time < sa_deadline and T > min_T:
        # Calculate subgradient direction
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        G = np.zeros_like(centers)
        eps = 1e-4
        
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    vec = centers[i] - centers[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        for i in range(N):
            if dist_to_walls[i] < r + eps:
                for k in range(3):
                    if centers[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0
        
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-9:
            G /= norm_G
        
        # Propose move
        step_size = 0.01 * (T / 0.02) + 0.002
        noise_scale = 0.01 * (T / 0.02)
        noise = np.random.normal(0, noise_scale, centers.shape)
        new_centers = centers + step_size * G + noise
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = min_radius(new_centers)
        
        delta = new_r - r
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        
        T *= cooling_rate
        
        # Occasionally restart if stuck?
        # If T is very low and we haven't improved in a while, maybe inject a larger kick?
        # For now, standard SA.
        
    # --- Phase 2: Local Optimization (Scipy SLSQP) ---
    # Use the best centers found so far as initial guess for a more precise local search.
    # SLSQP can handle the non-smoothness somewhat if we are careful, but it's designed for smooth problems.
    # We can formulate it as maximizing r subject to constraints, or minimizing -r.
    # However, scipy's minimize might get stuck in local minima or be slow.
    # Let's try a few random restarts of a local optimizer starting from perturbations of the best solution.
    
    local_deadline = time_limit - 5.0
    
    # Objective function for scipy: minimize -min_radius
    def objective(x):
        c = x.reshape(N, 3)
        # Ensure inside cube (scipy might explore outside)
        c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
        r = min_radius(c)
        return -r
    
    # Constraints: None explicitly needed if we clip, but SLSQP works better with bounds.
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    # Try a few local optimizations
    num_local_trials = 5
    for i in range(num_local_trials):
        if time.time() - start_time > local_deadline:
            break
        
        # Start from best solution + small random perturbation
        x0 = best_centers.copy()
        if i > 0:
            x0 += np.random.normal(0, 0.01, x0.shape)
            x0 = np.clip(x0, -0.5 + 1e-9, 0.5 - 1e-9)
        
        x0_flat = x0.flatten()
        
        try:
            res = minimize(objective, x0_flat, method='SLSQP', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                c_opt = res.x.reshape(N, 3)
                c_opt = np.clip(c_opt, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = min_radius(c_opt)
                if r_opt > best_r:
                    best_r = r_opt
                    best_centers = c_opt.copy()
        except Exception:
            pass

    # --- Phase 3: Final Polish with Gradient Ascent ---
    # Run a final, very fine-grained gradient ascent to squeeze out last bits of radius.
    polish_deadline = time_limit - 1.0
    centers = best_centers.copy()
    step = 0.001
    
    while time.time() - start_time < polish_deadline:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        r = min(np.min(d) / 2.0, np.min(dist_to_walls))
        
        G = np.zeros_like(centers)
        eps = 1e-5
        
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    vec = centers[i] - centers[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        for i in range(N):
            if dist_to_walls[i] < r + eps:
                for k in range(3):
                    if centers[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0
        
        norm_G = np.linalg.norm(G)
        if norm_G < 1e-9:
            break # Converged
            
        G /= norm_G
        
        # Adaptive step size
        new_centers = centers + step * G
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        new_r = min_radius(new_centers)
        
        if new_r > r:
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
            # Keep step or increase slightly?
            # step = min(step * 1.1, 0.005)
        else:
            step *= 0.5
            if step < 1e-8:
                break

    return best_centers
# EVOLVE-BLOCK-END
