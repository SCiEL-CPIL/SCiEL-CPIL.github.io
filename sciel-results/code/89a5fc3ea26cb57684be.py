"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import random
import math


def _hex_init(n):
    """Generate a hexagonal lattice initialization for n circles."""
    positions = []
    # Approximate optimal hex packing for 26 circles in a square
    # We'll generate a grid and pick the best n points
    # Or just use a structured grid that approximates hex
    
    # Simple hex-like grid
    # Rows of circles, alternating offset
    # Let's try to fit 5x5 or similar
    # For 26 circles, a 5x5 grid is 25, add 1? Or 6x5=30?
    
    # Let's use a jittered grid which is more robust for random restarts
    # But for the first restart, let's try a specific hex arrangement
    
    # Actually, for N=26, a good starting point is a 5x5 grid (25) + 1 in center?
    # Or just a 5x6 grid (30) and remove 4?
    
    # Let's stick to a robust jittered grid for all restarts to keep code simple,
    # but initialize with a slightly larger spacing to avoid immediate heavy overlaps.
    
    side = int(math.ceil(math.sqrt(n)))
    # Adjust side to get close to n
    # For 26, sqrt is ~5.1. side=5 gives 25. side=6 gives 36.
    # Let's use side=5 and add extra points if needed, or just use side=5 and accept 25? No, must be 26.
    # Let's use a 5x6 grid and randomly remove 4?
    # Or better: generate a grid of size side x side, and if we need more, add random points.
    
    # Strategy: Create a grid of size ceil(sqrt(n)).
    # If n is 26, side is 5. Grid has 25 points. We need 1 more.
    # We can split one cell or add a point in the center of the square if empty?
    # Easier: Just use a 6x6 grid (36 points) and randomly select 26? No, that's bad clustering.
    
    # Best: Use a 5x5 grid (25 points) and add 1 point at a random location far from others?
    # Or use a 5x6 grid and remove 4 random points?
    
    # Let's try a 5x6 grid (30 points) and remove 4 random points to get 26.
    # This maintains good spacing.
    rows = 5
    cols = 6
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) / cols
            y = (r + 0.5) / rows
            if r % 2 == 1:
                x += 0.5 / cols
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pts.append([x, y])
    
    # We have 30 points. We need 26.
    # Randomly select 26 points?
    # Better: Keep the ones closest to the center? Or just random.
    # Random selection from a hex grid is a decent restart.
    indices = random.sample(range(len(pts)), n)
    return np.array([pts[i] for i in indices])


def construct():
    """
    Pack 26 circles in the unit square [0,1]^2 with disjoint interiors,
    maximizing the sum of their radii.
    
    Uses a vectorized force-directed relaxation method.
    """
    
    best_solution = None
    best_sum = -1.0
    
    n_restarts = 10
    max_iterations = 2000
    dt = 0.02
    damping = 0.8
    
    for restart in range(n_restarts):
        # Initialize positions
        if restart == 0:
            # Specific hex-like initialization
            positions = _hex_init(N)
        else:
            # Jittered grid
            side = 5
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) < N:
                        x = (i + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        y = (j + 0.5) / side + random.uniform(-0.1, 0.1) / side
                        positions.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)])
            # If we need more (26 > 25), add one random
            if len(positions) < N:
                positions.append([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)])
            positions = np.array(positions)
        
        positions = np.array(positions, dtype=np.float64)
        velocities = np.zeros_like(positions)
        
        # Start with small equal radii
        radii = np.full(N, 0.05)
        
        # Vectorized force-directed relaxation
        for iteration in range(max_iterations):
            # Compute pairwise distances
            # diff[i, j] = pos[i] - pos[j]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            
            # Avoid division by zero
            dist = np.sqrt(dist_sq)
            np.fill_diagonal(dist, 1.0) # Temporary fix for diagonal, will zero out later
            
            # Compute overlap
            # min_dist[i, j] = radii[i] + radii[j]
            min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            
            # Overlap depth (positive if overlapping)
            overlap = min_dist - dist
            overlap = np.maximum(overlap, 0)
            
            # Force magnitude
            # F = k * overlap
            k = 50.0
            force_mag = k * overlap
            
            # Direction (unit vector)
            # We need to divide diff by dist
            # dist is zero on diagonal, we set it to 1.0 above
            unit_vec = diff / dist[:, :, np.newaxis]
            
            # Force vector on i due to j: force_mag[i,j] * unit_vec[i,j]
            # Sum over j
            # Note: unit_vec[i,j] points from j to i.
            # Force on i is repulsive, so in direction of unit_vec[i,j].
            forces = np.sum(force_mag[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Zero out diagonal contribution (should be 0 anyway since overlap=0)
            # But just in case
            np.fill_diagonal(force_mag, 0) # This doesn't affect forces sum correctly if not careful
            # Actually, the calculation above is correct. overlap[i,i] = 2*r - 0? No, dist[i,i]=0.
            # We set dist[i,i]=1.0. min_dist[i,i] = 2*r. overlap = 2*r - 1.
            # This is wrong. We must ensure diagonal is ignored.
            
            # Let's re-calculate forces more carefully to avoid diagonal issues.
            forces = np.zeros_like(positions)
            
            # Mask for i != j
            mask = ~np.eye(N, dtype=bool)
            
            # Compute forces only for i != j
            # dist is already computed
            # overlap is already computed
            # We need to zero out diagonal in overlap
            np.fill_diagonal(overlap, 0)
            
            # force_mag is k * overlap
            force_mag = k * overlap
            np.fill_diagonal(force_mag, 0)
            
            # unit_vec = diff / dist[:, :, np.newaxis]
            # dist diagonal is 1.0, so unit_vec diagonal is 0/1 = 0.
            # So forces = sum(force_mag * unit_vec) is safe.
            
            forces = np.sum(force_mag[:, :, np.newaxis] * unit_vec, axis=1)
            
            # Boundary forces
            # Left: x < r -> push right
            # Right: x > 1-r -> push left
            # Bottom: y < r -> push up
            # Top: y > 1-r -> push down
            
            # Left
            left_viol = np.maximum(radii - positions[:, 0], 0)
            forces[:, 0] += k * left_viol
            
            # Right
            right_viol = np.maximum(positions[:, 0] - (1 - radii), 0)
            forces[:, 0] -= k * right_viol
            
            # Bottom
            bottom_viol = np.maximum(radii - positions[:, 1], 0)
            forces[:, 1] += k * bottom_viol
            
            # Top
            top_viol = np.maximum(positions[:, 1] - (1 - radii), 0)
            forces[:, 1] -= k * top_viol
            
            # Update velocities and positions
            velocities = damping * velocities + forces * dt
            positions += velocities * dt
            
            # Clamp positions
            positions[:, 0] = np.clip(positions[:, 0], radii, 1 - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, 1 - radii)
            
            # Grow radii
            # Check max possible radius for each circle
            # 1. Distance to boundary
            max_r_boundary = np.minimum(
                np.minimum(positions[:, 0], 1 - positions[:, 0]),
                np.minimum(positions[:, 1], 1 - positions[:, 1])
            )
            
            # 2. Distance to other circles
            # dist[i,j] is already computed.
            # max_r[i] = min_j (dist[i,j] - radii[j]) / 2
            # We need to exclude j=i
            # dist_diag is 0 (or 1 if we didn't reset). Let's use the original dist_sq.
            # dist is sqrt(dist_sq). Diagonal is 0.
            # (dist - radii[:, np.newaxis]) / 2
            # For i=j, dist=0, radii[i]. Value = -radii[i]/2.
            # We want min over j != i.
            
            # Create a matrix of available radii
            avail_r = (dist - radii[np.newaxis, :]) / 2.0
            # Set diagonal to infinity so it doesn't affect min
            np.fill_diagonal(avail_r, np.inf)
            max_r_inter = np.min(avail_r, axis=1)
            
            max_r = np.minimum(max_r_boundary, max_r_inter)
            
            # Grow radii towards max_r
            # If current radius < max_r, increase it
            # Use a small step size
            growth = 0.001
            new_radii = np.minimum(radii + growth, max_r)
            
            # Only update if it helps? Or just always update?
            # If we increase radii, we might create overlaps.
            # The force simulation will resolve them.
            radii = new_radii
            
            # Check convergence
            if iteration > 100:
                # Check if forces are small
                max_force = np.max(np.abs(forces))
                if max_force < 1e-4:
                    break
        
        # Validate and adjust
        # Ensure strict non-overlap
        # We might have slight overlaps due to numerical precision or aggressive growth
        # Perform a final cleanup pass
        
        valid = True
        for i in range(N):
            if radii[i] < 0:
                valid = False
                break
            if positions[i, 0] < radii[i] or positions[i, 0] > 1 - radii[i]:
                valid = False
                break
            if positions[i, 1] < radii[i] or positions[i, 1] > 1 - radii[i]:
                valid = False
                break
        
        if valid:
            # Check pairwise
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist = radii[i] + radii[j]
                    if dist_sq < min_dist * min_dist:
                        # Slight overlap, shrink radii or move?
                        # For scoring, we need disjoint interiors.
                        # Shrink the smaller one? Or both?
                        # Let's shrink both by half the overlap
                        overlap = min_dist - math.sqrt(dist_sq)
                        if overlap > 0:
                            radii[i] -= overlap / 2
                            radii[j] -= overlap / 2
                            # Ensure radii don't go negative
                            if radii[i] < 0: radii[i] = 0
                            if radii[j] < 0: radii[j] = 0
                            valid = False # Mark as needing re-check? 
                            # Actually, if we shrink, it might fix it.
                            # But we modified radii. Let's just accept it if it's valid after shrinking.
            
            # Re-validate after potential shrinking
            valid = True
            for i in range(N):
                if radii[i] < 0:
                    valid = False
                    break
                if positions[i, 0] < radii[i] - 1e-6 or positions[i, 0] > 1 - radii[i] + 1e-6:
                    valid = False
                    break
                if positions[i, 1] < radii[i] - 1e-6 or positions[i, 1] > 1 - radii[i] + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx * dx + dy * dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist * min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                total = np.sum(radii)
                if total > best_sum:
                    best_sum = total
                    best_solution = np.column_stack([positions[:, 0], positions[:, 1], radii])
    
    # Fallback
    if best_solution is None:
        # Simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
        # Add one more if needed
        if len(cells) < N:
            cells.append([0.5, 0.5, 0.01])
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
