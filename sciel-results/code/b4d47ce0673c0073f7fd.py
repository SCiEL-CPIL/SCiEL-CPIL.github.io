"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_score(centres):
    """Compute the valid radius for a given set of centres."""
    # Check validity
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres[:, 0] < -0.5) or np.any(centres[:, 0] > 0.5) or np.any(centres[:, 1] < -0.5) or np.any(centres[:, 1] > 0.5):
        return 0.0
    
    # Min distance to walls
    min_wall = np.min(np.min(np.abs(centres), axis=1) - 0.0) # Wait, distance from edge
    # Distance from centre to wall is 0.5 - |coord|
    dist_to_walls = np.minimum(0.5 - np.abs(centres[:, 0]), 0.5 - np.abs(centres[:, 1]))
    r_wall = np.min(dist_to_walls)
    
    # Min distance between circles
    # To be efficient, we can just compute full pairwise for N=200 (40k pairs)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r_pair = min_pair / 2.0
    
    return min(r_wall, r_pair)

def _soft_sphere_relax(centres, steps=2000, lr_start=0.01, lr_end=0.0001, repulsion_power=4):
    """
    Relax circles using repulsive forces.
    Force is proportional to (1/d)^repulsion_power.
    """
    # Ensure we work on a copy
    c = centres.copy()
    n = len(c)
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for step in range(steps):
        # Decay learning rate
        lr = lr_start * (lr_end / lr_start) ** (step / steps)
        
        # Compute distances for all pairs
        diff = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-9)
        
        # Repulsive force magnitude
        # We want to push them apart.
        # Force vector on i is +diff/d * magnitude
        # Force vector on j is -diff/d * magnitude
        forces = (1.0 / dists) ** repulsion_power
        force_vectors = diff / dists[:, np.newaxis] * forces[:, np.newaxis]
        
        # Accumulate forces
        disp = np.zeros_like(c)
        np.add.at(disp, i_idx, force_vectors)
        np.add.at(disp, j_idx, -force_vectors)
        
        # Wall repulsion
        # Push circles away from walls
        # x walls: if c_x < 0, push +x; if c_x > 0, push -x
        # Distance to wall: 0.5 - |c_x|
        dist_x = 0.5 - np.abs(c[:, 0])
        dist_y = 0.5 - np.abs(c[:, 1])
        
        # If close to wall, push strongly
        # Threshold
        thresh = 0.05
        wall_forces_x = np.zeros_like(c[:, 0])
        wall_forces_y = np.zeros_like(c[:, 1])
        
        # For x
        mask_x = dist_x < thresh
        if np.any(mask_x):
            # Force direction: away from wall
            # If c_x < 0, push +x (sign is +1)
            # If c_x > 0, push -x (sign is -1)
            # Magnitude: (1/dist)^power
            d = np.maximum(dist_x[mask_x], 1e-9)
            mag = (1.0 / d) ** repulsion_power
            signs = np.sign(c[mask_x, 0])
            # If exactly 0, sign is 0, but we don't want 0 force. 
            # Randomize or just push +x if 0? 
            # Let's say if 0, push +x.
            signs[signs == 0] = 1.0
            wall_forces_x[mask_x] = signs * mag
            
        # For y
        mask_y = dist_y < thresh
        if np.any(mask_y):
            d = np.maximum(dist_y[mask_y], 1e-9)
            mag = (1.0 / d) ** repulsion_power
            signs = np.sign(c[mask_y, 1])
            signs[signs == 0] = 1.0
            wall_forces_y[mask_y] = signs * mag
            
        disp[:, 0] += wall_forces_x
        disp[:, 1] += wall_forces_y
        
        # Update positions
        c += lr * disp
        
        # Clip to square (with small margin to avoid sticking to edge in a way that breaks differentiability if we were using grad, but here we just clip)
        # Actually, clipping can cause "stickiness". 
        # Better to rely on wall forces. But for safety, clip to [-0.5, 0.5].
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        
    return c

def _bottleneck_optimize(centres, max_iters=1000):
    """
    Local optimization targeting the minimum radius.
    1. Find the bottleneck (min pair or min wall).
    2. Move the involved circles to increase this minimum.
    3. Check if global min improved.
    """
    c = centres.copy()
    n = len(c)
    best_r = _compute_score(c)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iters):
        # Compute current min radius components
        diff = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        r_pair = min_pair_dist / 2.0
        
        dist_to_walls = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        min_wall_idx = np.argmin(dist_to_walls)
        min_wall_dist = dist_to_walls[min_wall_idx]
        
        current_r = min(r_pair, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            # We could update c here, but we are modifying c in place.
            # We just need to ensure we don't degrade.
            
        # Determine which is the bottleneck
        if r_pair < min_wall_dist:
            # Pair is bottleneck
            i = i_idx[min_pair_idx]
            j = j_idx[min_pair_idx]
            
            # Move i and j apart along their connecting vector
            # Step size: small fraction of the gap
            # We want to increase dist by some amount.
            # Let's try moving each by 0.5 * step_size
            # Step size should be small to not violate other constraints.
            # Heuristic: step = 0.01 * current_r or fixed small step?
            # Let's use a step that tries to close 10% of the gap to the "ideal" distance?
            # No, just push apart.
            
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Coincident, randomize
                c[i] += np.random.normal(0, 0.01, 2)
                c[j] -= np.random.normal(0, 0.01, 2)
            else:
                unit_vec = vec / norm
                # Move i by +unit_vec * step, j by -unit_vec * step
                # Step size: 0.005
                step = 0.005
                c[i] += unit_vec * step
                c[j] -= unit_vec * step
        else:
            # Wall is bottleneck
            i = min_wall_idx
            # Determine which wall
            x = c[i, 0]
            y = c[i, 1]
            
            # Distances to each wall
            d_left = x + 0.5
            d_right = 0.5 - x
            d_bottom = y + 0.5
            d_top = 0.5 - y
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            
            step = 0.005
            
            if min_d == d_left:
                c[i, 0] += step
            elif min_d == d_right:
                c[i, 0] -= step
            elif min_d == d_bottom:
                c[i, 1] += step
            else:
                c[i, 1] -= step
        
        # Clip
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        
        # Check if we made progress in the last few steps? 
        # For simplicity, just run fixed iterations.
        
    return c

def _generate_hex_grid():
    """Generate a hexagonal packing initialization."""
    # Hexagonal packing in a square is tricky to fit exactly N.
    # We'll generate a dense hex grid and select the best N points that fit.
    # Or just generate a grid and jitter.
    
    # Let's try a standard hex grid.
    # Spacing s.
    # Rows are offset.
    # Number of rows R, cols C.
    # Approx N = R * C.
    
    # Let's just use a square grid with jitter for simplicity and robustness,
    # as hex grid alignment with square boundary can be wasteful.
    # But hex is better density.
    
    # Let's generate a hex grid that covers the square.
    # s = 1.0 / sqrt(N) approx.
    # For hex, area per circle is s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 = 1 => s = sqrt(2 / (N * sqrt(3)))
    
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points
    pts = []
    # y step
    dy = s * np.sqrt(3.0) / 2.0
    # x step
    dx = s
    
    # Number of rows
    # Square is 1x1.
    # y from -0.5 to 0.5
    num_rows = int(np.ceil(1.0 / dy)) + 1
    # Offset
    for row in range(num_rows):
        y = -0.5 + row * dy
        if y > 0.5:
            break
        # Offset x for odd rows
        offset = 0.0
        if row % 2 != 0:
            offset = dx / 2.0
            
        # x from -0.5 to 0.5
        num_cols = int(np.ceil(1.0 / dx)) + 1
        for col in range(num_cols):
            x = -0.5 + col * dx + offset
            if x > 0.5:
                break
            pts.append([x, y])
            
    pts = np.array(pts)
    
    # If we have more than N, select N that maximize min distance?
    # Or just take first N?
    # Taking first N might bias to top-left.
    # Let's center the selection.
    # Or just random sample if too many?
    # Better: sort by distance from center?
    
    if len(pts) > N:
        # Calculate distance to center (0,0)
        dists = np.sum(pts**2, axis=1)
        # Sort by distance, take N closest to center
        indices = np.argsort(dists)
        pts = pts[indices[:N]]
    elif len(pts) < N:
        # If we don't have enough, we need to generate more or adjust s.
        # This shouldn't happen with the +1 padding usually.
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # 1. Generate initial hex grid
    c = _generate_hex_grid()
    
    # 2. Soft sphere relaxation
    c = _soft_sphere_relax(c, steps=1000)
    
    current_score = _compute_score(c)
    
    # 3. Bottleneck optimization
    # Run multiple rounds of bottleneck optimization with different step sizes?
    # Or just one long run.
    
    # We can interleave relaxation and bottleneck.
    # Let's do:
    # Relax -> Bottleneck -> Relax -> Bottleneck
    
    while time.time() < deadline:
        # Bottleneck step
        c = _bottleneck_optimize(c, max_iters=200)
        s1 = _compute_score(c)
        
        # Relaxation step (shorter)
        c = _soft_sphere_relax(c, steps=500, lr_start=0.005, lr_end=0.0001)
        s2 = _compute_score(c)
        
        # Keep the best
        if s2 > current_score:
            current_score = s2
        elif s1 > current_score:
             # If relaxation degraded it, we should probably revert to s1 state?
             # But we modified c in place. 
             # We didn't save s1 state.
             # Let's just accept c from relaxation. If it's worse, the next bottleneck step might fix it.
             pass
             
    return c

# EVOLVE-BLOCK-END
