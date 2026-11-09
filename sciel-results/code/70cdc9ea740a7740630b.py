"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Maximize sum of radii of N=21 disjoint circles in a rectangle with perimeter 4.
    Strategy:
    1. Iterate over candidate aspect ratios.
    2. For each aspect ratio, perform multi-start optimization.
    3. Optimization: Iteratively increase radii and resolve overlaps by pushing circles apart.
    4. Use vectorized NumPy operations for speed to fit within time limits.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_circles = None
    best_ar = 1.0

    # Candidate aspect ratios (w/h). Perimeter 4 => 2(w+h)=4 => w+h=2.
    # w = 2*ar/(1+ar), h = 2/(1+ar)
    # Square is 1.0. Range [0.5, 2.0] covers significant variation.
    aspect_ratios = [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0
    ]
    
    # Shuffle aspect ratios to prioritize diverse shapes early, or keep sorted?
    # Sorted is fine, but randomizing might help if some aspect ratios are traps.
    # Let's stick to a sensible order but maybe interleave.
    
    # Initial grid configuration in normalized [0,1]x[0,1] space
    side = int(np.ceil(np.sqrt(N)))
    grid_x = []
    grid_y = []
    for i in range(side):
        for j in range(side):
            if len(grid_x) < N:
                grid_x.append((i + 0.5) / side)
                grid_y.append((j + 0.5) / side)
    init_x = np.array(grid_x)
    init_y = np.array(grid_y)

    # Optimization parameters
    max_relax_steps = 300  # Relaxation steps per radius growth iteration
    radius_growth_step = 0.005
    max_radius = 0.5
    relax_iters = 50  # Total iterations of (Grow + Relax)
    
    # We need to be careful with time.
    # N=21, pairs=210. Vectorized ops are fast.
    # Let's estimate: 14 ARs * 10 restarts * 50 relax_iters * 300 steps * 210 pairs...
    # This might be too slow if not optimized.
    # Vectorized: 14 * 10 * 50 * 300 * (ops on 21x21 matrix)
    # 14*10*50*300 = 2.1M iterations.
    # Each iteration: compute distances (21x21), check overlaps, apply forces.
    # 2.1M * ~100ns (vectorized) ~ 0.2s. This is very fast.
    # We can afford more restarts or steps.
    
    restarts_per_ar = 15
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        for restart in range(restarts_per_ar):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize positions
            # Use grid for restart 0, random for others
            if restart == 0:
                x = init_x * w
                y = init_y * h
            else:
                # Random initialization, scaled to fit
                # Ensure they don't start overlapping heavily
                x = np.random.uniform(0.1, 0.9, N) * w
                y = np.random.uniform(0.1, 0.9, N) * h
            
            # Initial radii: small
            r = np.full(N, 0.05)
            
            current_sum_r = np.sum(r)
            
            # Optimization Loop
            for grow_iter in range(relax_iters):
                if time.time() > deadline - 2.0:
                    break
                
                # 1. Grow radii
                r[:] += radius_growth_step
                # Cap radii
                r[:] = np.minimum(r, max_radius)
                
                # 2. Relax positions to resolve overlaps and fit in box
                # We do multiple small relaxation steps
                for step in range(max_relax_steps):
                    # Compute pairwise distances
                    # dx[i,j] = x[i] - x[j]
                    dx = x[:, None] - x[None, :]
                    dy = y[:, None] - y[None, :]
                    dist_sq = dx**2 + dy**2
                    # Avoid division by zero
                    dist_sq[dist_sq < 1e-12] = 1e-12
                    dist = np.sqrt(dist_sq)
                    
                    # Required separation
                    min_dist = r[:, None] + r[None, :]
                    
                    # Overlap amount (positive if overlapping)
                    overlap = min_dist - dist
                    # Only consider overlaps
                    # Mask for i < j to avoid double counting forces?
                    # Forces are symmetric. We can compute force on i from j.
                    # F_ij is along (x_i - x_j).
                    # If dist < min_dist, push apart.
                    
                    # Normalized direction
                    # Handle dist ~ 0
                    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
                    nx = dx / dist_safe
                    ny = dy / dist_safe
                    
                    # Force magnitude proportional to overlap
                    # Only apply if overlapping
                    force_mag = np.maximum(0, overlap)
                    
                    # Sum of forces on each circle
                    # F_i = sum_j F_ij
                    # F_ij vector = force_mag[i,j] * (nx[i,j], ny[i,j])
                    # Note: nx[i,j] points from j to i? 
                    # dx[i,j] = x[i] - x[j]. So nx points from j to i.
                    # If i and j overlap, i should move away from j (direction +nx)
                    # and j should move away from i (direction -nx).
                    
                    fx = np.sum(force_mag * nx, axis=1)
                    fy = np.sum(force_mag * ny, axis=1)
                    
                    # Wall constraints
                    # Push in if outside
                    # Left: x < r => force +
                    # Right: x > w-r => force -
                    # Bottom: y < r => force +
                    # Top: y > h-r => force -
                    
                    fx_wall = np.zeros(N)
                    fy_wall = np.zeros(N)
                    
                    # Vectorized wall forces
                    # Left
                    mask_l = x < r
                    fx_wall[mask_l] += (r[mask_l] - x[mask_l])
                    # Right
                    mask_r = x > w - r
                    fx_wall[mask_r] -= (x[mask_r] - (w - r[mask_r]))
                    # Bottom
                    mask_b = y < r
                    fy_wall[mask_b] += (r[mask_b] - y[mask_b])
                    # Top
                    mask_t = y > h - r
                    fy_wall[mask_t] -= (y[mask_t] - (h - r[mask_t]))
                    
                    # Total force
                    fx_total = fx + fx_wall
                    fy_total = fy + fy_wall
                    
                    # Update positions
                    # Simple gradient ascent/descent step
                    # Step size needs to be small to avoid oscillation
                    step_size = 0.5
                    x += fx_total * step_size
                    y += fy_total * step_size
                    
                    # Hard clamp to prevent flying out
                    x = np.clip(x, r, w - r)
                    y = np.clip(y, r, h - r)
                
                # Check validity and update best
                # A configuration is valid if no overlaps.
                # We just checked in the last relaxation step, but let's verify.
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist_sq = dx**2 + dy**2
                min_dist = r[:, None] + r[None, :]
                
                # Check if any pair overlaps (dist < min_dist)
                # dist_sq < min_dist^2
                # Exclude diagonal
                np.fill_diagonal(dist_sq, np.inf)
                np.fill_diagonal(min_dist, np.inf)
                
                # If min(dist_sq) > max(min_dist^2) roughly?
                # Actually, we need min_dist_sq > dist_sq for all pairs.
                # So we need: min(dist_sq - min_dist^2) > 0 ?
                # Or simply: np.any(dist_sq < min_dist**2) should be False.
                
                # To be efficient:
                # overlap_mask = dist_sq < min_dist**2
                # if not np.any(overlap_mask):
                
                # Let's compute the maximum violation
                # violation = min_dist**2 - dist_sq
                # max_violation = np.max(violation)
                # if max_violation < 1e-6: # Valid
                
                violation = min_dist**2 - dist_sq
                max_violation = np.max(violation)
                
                if max_violation < 1e-5:
                    sum_r = np.sum(r)
                    if sum_r > best_sum_r:
                        best_sum_r = sum_r
                        best_circles = np.column_stack((x, y, r))
                        best_ar = ar
                        
                        # Early termination if we hit a theoretical bound?
                        # Area bound: N * pi * r_avg^2 <= w*h
                        # r_avg <= sqrt(w*h / (N*pi))
                        # Sum r <= N * sqrt(w*h / (N*pi)) = sqrt(N * w*h / pi)
                        # For w=1, h=1: sqrt(21/pi) ~ 2.58
                        # Current best might be lower.
                        
    # If no valid configuration was found (should not happen), return a safe grid
    if best_circles is None:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    # Use a square 1x1 box (perimeter 4)
                    r = 0.1
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    cells.append([x, y, r])
        best_circles = np.array(cells)
        
    return best_circles
# EVOLVE-BLOCK-END
