"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(centers):
    """Compute the valid radius for a given set of centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    min_dist_to_wall = np.min(0.5 - np.abs(centers))
    if min_dist_to_wall < 0:
        return 0.0
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

def _initial_hex_grid():
    """Generate a hexagonal lattice configuration for N points."""
    # A hex grid with rows of 14 and 15 points fits 200 points (14*14 + 15*14? No, 14+15+... let's just generate and crop)
    n_rows = 14
    n_cols = 15
    d = 1.0 / (n_cols - 0.5)
    row_spacing = d * math.sqrt(3) / 2.0
    
    pts = []
    for row in range(n_rows):
        y = -0.5 + d/2.0 + row * row_spacing
        if row % 2 == 0:
            n_in_row = n_cols
            x_start = -0.5 + d/2.0
        else:
            n_in_row = n_cols - 1
            x_start = -0.5 + d
            
        for col in range(n_in_row):
            x = x_start + col * d
            pts.append([x, y])
            
    pts = np.array(pts)
    if len(pts) >= N:
        pts = pts[:N]
    else:
        # Add remaining points randomly if needed (shouldn't happen with these dims)
        while len(pts) < N:
            idx = np.random.randint(len(pts))
            new_pt = pts[idx] + np.random.randn(2) * d * 0.1
            new_pt = np.clip(new_pt, -0.5, 0.5)
            pts = np.vstack([pts, new_pt])
            
    return pts

def _md_relaxation(centers, max_iter=20000, dt=0.001, cooling_rate=0.9995, deadline=None):
    """
    Molecular Dynamics relaxation.
    Treats circles as particles with repulsive forces when too close,
    and attractive forces to center/walls when too far (or just repulsive from walls).
    Actually, for packing, we want to maximize the minimum distance.
    A common trick is to minimize a potential energy E = sum(1/d_ij) + sum(1/d_wall) 
    or similar, but that's tricky.
    
    Better: Simple iterative relaxation.
    1. Find the closest pair (i, j) and the point closest to a wall.
    2. Move i and j apart slightly.
    3. Move wall-point away from wall slightly.
    4. Repeat.
    
    This is essentially a targeted local search but framed as physics.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Precompute indices for faster access? No, N is small enough (200).
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # 1. Find closest pair
        min_dist = np.inf
        min_i, min_j = -1, -1
        min_wall_dist = np.inf
        min_wall_i = -1
        min_wall_axis = 0
        min_wall_sign = 0
        
        # O(N^2) distance check
        for i in range(n):
            # Wall checks
            for k in range(2):
                # Distance to left wall (x = -0.5)
                d_left = centers[i][k] - (-0.5) if k==0 else centers[i][k] - (-0.5) # Wait, axis 0 is x, axis 1 is y
                # Let's be explicit
                if k == 0:
                    d_left = centers[i][0] + 0.5
                    d_right = 0.5 - centers[i][0]
                    if d_left < min_wall_dist:
                        min_wall_dist = d_left
                        min_wall_i = i
                        min_wall_axis = 0
                        min_wall_sign = -1 # Push right (+x)
                    if d_right < min_wall_dist:
                        min_wall_dist = d_right
                        min_wall_i = i
                        min_wall_axis = 0
                        min_wall_sign = 1 # Push left (-x)
                else:
                    d_bottom = centers[i][1] + 0.5
                    d_top = 0.5 - centers[i][1]
                    if d_bottom < min_wall_dist:
                        min_wall_dist = d_bottom
                        min_wall_i = i
                        min_wall_axis = 1
                        min_wall_sign = -1 # Push up (+y)
                    if d_top < min_wall_dist:
                        min_wall_dist = d_top
                        min_wall_i = i
                        min_wall_axis = 1
                        min_wall_sign = 1 # Push down (-y)
            
            # Pair checks
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j
        
        if min_i == -1:
            break
            
        # Determine move magnitude
        # Adaptive step: smaller as we get closer to optimal
        # Target radius is roughly 0.036. Current min_dist/2 is current radius.
        # We want to increase min_dist.
        
        # Move closest pair apart
        if min_dist < 0.1: # Only adjust if they are somewhat close
            diff = centers[min_i] - centers[min_j]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                direction = diff / norm
                # Move each by half the step
                step = 0.0005 # Small step
                # Damping based on iteration
                step *= (1.0 - it / max_iter) * 2.0
                step = max(step, 0.00001)
                
                new_i = centers[min_i] + direction * step
                new_j = centers[min_j] - direction * step
                
                # Check boundaries
                if np.all(np.abs(new_i) <= 0.5) and np.all(np.abs(new_j) <= 0.5):
                    centers[min_i] = new_i
                    centers[min_j] = new_j
        
        # Move wall-violating point away from wall
        # We use the global min_wall_dist found above
        # But we should prioritize the point with the smallest wall distance
        # The logic above already found the single smallest wall distance.
        
        if min_wall_dist < 0.04: # Threshold
            i = min_wall_i
            axis = min_wall_axis
            sign = min_wall_sign
            
            step = 0.0005
            step *= (1.0 - it / max_iter) * 2.0
            step = max(step, 0.00001)
            
            new_pos = centers[i].copy()
            new_pos[axis] += sign * step
            
            if np.all(np.abs(new_pos) <= 0.5):
                centers[i] = new_pos
                
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with hex grid
    centers = _initial_hex_grid()
    
    # Run MD relaxation
    centers = _md_relaxation(centers, max_iter=30000, deadline=deadline - 10)
    
    # Final polish with a few more iterations if time permits
    if time.time() < deadline - 2:
        centers = _md_relaxation(centers, max_iter=10000, deadline=deadline - 1)
        
    return centers
# EVOLVE-BLOCK-END
