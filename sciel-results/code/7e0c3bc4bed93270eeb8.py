"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    n = N  # 50

    def compute_radius(centers):
        """Compute the max common radius for given centers (vectorized)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def relax(centers, r, steps=2000, dt=0.001):
        """Relax circles to avoid overlap at radius r using force-directed method."""
        centers = centers.copy()
        # Compute initial forces
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs closer than 2*r + margin
            mask = dists < 2.0 * r + 0.01
            mask = mask & (dists > 1e-10)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i, j]:
                        d = centers[i] - centers[j]
                        dist = dists[i, j]
                        # Repulsive force: strong when overlapping
                        if dist < 2.0 * r:
                            # Force proportional to overlap
                            overlap = 2.0 * r - dist
                            f = overlap * 10.0  # spring constant
                        else:
                            f = 0.0
                        fvec = d / dist * f if dist > 1e-10 else np.zeros(2)
                        forces[i] += fvec
                        forces[j] -= fvec
            
            # Boundary forces: push circles back if they exceed bounds
            for i in range(n):
                for dim in range(2):
                    if centers[i, dim] > 0.5 - r:
                        forces[i, dim] -= (centers[i, dim] - (0.5 - r)) * 10.0
                    elif centers[i, dim] < -0.5 + r:
                        forces[i, dim] += ((-0.5 + r) - centers[i, dim]) * 10.0
            
            # Update positions
            centers += dt * forces
            # Clip to valid range
            centers = np.clip(centers, -0.5 + r, 0.5 - r)
        
        return centers

    def grow_radius(centers, r_init, r_max=0.1, steps=500):
        """Grow radius while relaxing to find optimal packing."""
        centers = centers.copy()
        r = r_init
        
        # Binary search style: try to increase radius
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Try to increase radius slightly
            r_new = r + 0.0001
            if r_new > r_max:
                break
            
            # Relax at new radius
            centers_relaxed = relax(centers, r_new, steps=100, dt=0.0005)
            
            # Check if packing is valid at r_new
            r_actual = compute_radius(centers_relaxed)
            if r_actual >= r_new - 1e-9:
                r = r_new
                centers = centers_relaxed
            else:
                # Couldn't grow, try smaller increment
                r_new = r + 0.00002
                if r_new <= r:
                    break
                centers_relaxed = relax(centers, r_new, steps=50, dt=0.0005)
                r_actual = compute_radius(centers_relaxed)
                if r_actual >= r_new - 1e-9:
                    r = r_new
                    centers = centers_relaxed
                else:
                    break
        
        return centers, r

    def init_hex():
        """Hexagonal lattice initialization."""
        pts = []
        best_pts = None
        best_r = 0
        
        for rows in range(5, 11):
            for cols in range(5, 11):
                if rows * cols < 50 or rows * cols > 55:
                    continue
                s_x = 1.0 / (cols - 0.5)
                s_y = 2.0 / ((rows - 1) * np.sqrt(3)) if rows > 1 else 1.0
                s = min(s_x, s_y)
                
                pts = []
                for r in range(rows):
                    for c in range(cols):
                        x = (c + (0.5 if r % 2 == 1 else 0)) * s - (cols - 0.5) * s / 2
                        y = r * s * np.sqrt(3) / 2 - (rows - 1) * s * np.sqrt(3) / 4
                        pts.append([x, y])
                
                if len(pts) >= 50:
                    pts = pts[:50]
                    pts_arr = np.array(pts)
                    if np.all(np.abs(pts_arr) <= 0.5):
                        r_test = compute_radius(pts_arr)
                        if r_test > best_r:
                            best_r = r_test
                            best_pts = pts_arr
        
        return best_pts if best_pts is not None else None

    def init_grid(rows, cols):
        """Simple grid initialization."""
        if rows * cols < 50:
            return None
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        return grid[:50] if len(grid) >= 50 else None

    # Generate initializations
    inits = []
    
    hex_pts = init_hex()
    if hex_pts is not None:
        inits.append(hex_pts)
    
    for rows, cols in [(7, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        g = init_grid(rows, cols)
        if g is not None:
            inits.append(g)
    
    # Add jittered versions
    np.random.seed(42)
    for base in inits[:3]:
        jittered = base + 0.01 * np.random.randn(50, 2)
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # Find best initial
    best_init = None
    best_init_r = 0
    for init in inits:
        r = compute_radius(init)
        if r > best_init_r:
            best_init_r = r
            best_init = init
    
    if best_init is None:
        ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
        best_init = np.array([(x, y) for y in ticks7 for x in ticks7])
        best_init = np.vstack([best_init, np.array([[0.0, 0.0]])])[:50]
    
    best_centers = best_init.copy()
    best_r = best_init_r
    
    # Try growing from multiple initializations
    for init_idx, init in enumerate(inits[:6]):
        if time.time() > deadline - 15:
            break
        
        # Start from a small radius and grow
        r_start = compute_radius(init) * 0.9
        centers_grown, r_grown = grow_radius(init, r_start, r_max=0.08, steps=300)
        
        r_final = compute_radius(centers_grown)
        if r_final > best_r:
            best_r = r_final
            best_centers = centers_grown.copy()
    
    # Final refinement: relax at best radius
    if time.time() < deadline - 5:
        centers_refined = relax(best_centers, best_r, steps=500, dt=0.0005)
        r_refined = compute_radius(centers_refined)
        if r_refined > best_r:
            best_r = r_refined
            best_centers = centers_refined
    
    # Safety check
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.499999, neginf=-0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
