import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_result = None
    best_score = -1
    
    # Try different aspect ratios
    # w + h = 2, so w in (0, 2), h = 2 - w
    # Focus on aspect ratios close to 1 and some variations
    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.83, 1.3, 0.77, 1.4, 0.71, 1.5, 0.67, 1.6, 0.62, 1.8, 0.56, 2.0, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))  # w + h = 2, w/h = ar
        h = 2.0 - w
        
        # Grid initialization
        # Choose grid dimensions based on aspect ratio
        if ar >= 1.0:
            cols = int(np.round(np.sqrt(N * ar)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.round(np.sqrt(N / ar)))
            cols = int(np.ceil(N / rows))
        
        # Ensure we have at least N cells
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Place N circles in a grid
        positions = []
        radii = []
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * w / cols
            y = (row + 0.5) * h / rows
            positions.append([x, y])
        
        positions = np.array(positions)
        
        # Initial radii: limited by spacing and boundaries
        # Spacing-based radius
        dx = w / cols
        dy = h / rows
        r_init = 0.5 * min(dx, dy) * 0.95
        
        # Also limited by boundaries
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            radii.append(min(r_init, r_max))
        
        radii = np.array(radii)
        
        # Physics-based relaxation
        # We use a vectorized approach for pairwise interactions
        # This is much faster than nested loops
        
        # Precompute indices for pairwise differences
        # i < j pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Compute pairwise distances
            dx = positions[j_idx, 0] - positions[i_idx, 0]
            dy = positions[j_idx, 1] - positions[i_idx, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            
            # Avoid division by zero
            dist[dist < 1e-8] = 1e-8
            
            # Minimum required distance
            min_dist = radii[i_idx] + radii[j_idx]
            
            # Overlap amount (positive if overlapping)
            overlap = min_dist - dist
            
            # Force calculation
            # Lennard-Jones like: Strong repulsion if overlapping, weak attraction if close
            # We want to push apart if overlapping.
            # Force magnitude proportional to overlap, but with a smooth cutoff or power law.
            # Let's use a simple linear repulsion with a damping factor.
            
            # Normalize direction
            dir_x = dx / dist
            dir_y = dy / dist
            
            # Repulsive force
            # Scale factor: strong enough to separate quickly, but stable
            force_mag = np.where(overlap > 0, overlap * 2.0, 0.0)
            
            # Add a small attractive force if very close but not overlapping? 
            # No, for packing, we just want to maximize radii, so we don't need attraction between circles.
            # We just need to resolve overlaps and expand radii.
            
            # Accumulate forces
            forces = np.zeros((N, 2))
            np.add.at(forces, i_idx, -force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
            np.add.at(forces, j_idx, force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
            
            # Boundary forces
            # Push away from boundaries if too close
            # Margin: 0.0
            # Force if x < r or x > w - r
            left_viol = np.maximum(0, radii - positions[:, 0])
            right_viol = np.maximum(0, positions[:, 0] + radii - w)
            bottom_viol = np.maximum(0, radii - positions[:, 1])
            top_viol = np.maximum(0, positions[:, 1] + radii - h)
            
            forces[:, 0] += (left_viol - right_viol) * 5.0
            forces[:, 1] += (bottom_viol - top_viol) * 5.0
            
            # Radius update
            # Try to expand radii
            # Max radius allowed by boundaries
            r_boundary = np.minimum(np.minimum(positions[:, 0], w - positions[:, 0]), 
                                    np.minimum(positions[:, 1], h - positions[:, 1]))
            
            # Max radius allowed by neighbors
            # For each circle i, min over j != i of (dist_ij - r_j)
            # We can compute this using the pairwise distances
            # dist_ij - r_j is the space available for r_i to grow
            # Note: dist is computed for pairs (i, j) with i < j.
            # We need to update a matrix or array of min distances.
            
            # Initialize max allowed radius by neighbors to infinity
            r_neighbor_max = np.full(N, np.inf)
            
            # For each pair (i, j):
            # space_for_i = dist - r_j
            # space_for_j = dist - r_i
            space_for_i = dist - radii[j_idx]
            space_for_j = dist - radii[i_idx]
            
            # Update min for i
            np.minimum.at(r_neighbor_max, i_idx, space_for_i)
            # Update min for j
            np.minimum.at(r_neighbor_max, j_idx, space_for_j)
            
            # Combined max radius
            r_max = np.minimum(r_boundary, r_neighbor_max)
            r_max = np.maximum(r_max, 0.0)
            
            # Grow radii towards r_max
            # Use a learning rate for radius growth
            # If r < r_max, increase it.
            dr = (r_max - radii) * 0.1
            
            # Apply updates
            # Adaptive step size
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 1e-6:
                step_size = 0.05 / (1.0 + max_force)
            else:
                step_size = 0.05
            
            positions += forces * step_size
            radii += dr * 0.5 # Slightly slower radius growth to maintain stability
            
            # Clamp
            radii = np.maximum(radii, 0.001)
            
            # Ensure positions are within bounds (hard constraint)
            # If a circle goes out, pull it back
            positions[:, 0] = np.clip(positions[:, 0], radii, w - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, h - radii)
            
            # Check validity and score
            # Valid if no overlaps and within bounds
            # We already enforced bounds via clipping, but check overlaps
            # Overlap check: dist < r_i + r_j
            # We have dist and min_dist from before, but they are stale.
            # Recalculate for scoring?
            # To save time, we can just check if any overlap > 1e-5
            # dist is from the start of the iteration. 
            # Let's do a quick check on the updated positions.
            # Actually, the forces should have resolved most overlaps.
            # Let's just compute the score if it looks valid.
            
            # Quick validity check
            valid = True
            # Check bounds (should be true due to clipping, but just in case)
            if np.any(positions[:, 0] < radii - 1e-8) or np.any(positions[:, 0] > w - radii + 1e-8):
                valid = False
            if np.any(positions[:, 1] < radii - 1e-8) or np.any(positions[:, 1] > h - radii + 1e-8):
                valid = False
                
            if valid:
                # Check overlaps
                # Re-calculate dist for current positions
                dx_c = positions[j_idx, 0] - positions[i_idx, 0]
                dy_c = positions[j_idx, 1] - positions[i_idx, 1]
                dist_c = np.sqrt(dx_c * dx_c + dy_c * dy_c)
                min_dist_c = radii[i_idx] + radii[j_idx]
                
                if np.any(dist_c < min_dist_c - 1e-8):
                    valid = False
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([positions, radii]).copy()
        
        # Final check for this aspect ratio
        valid = True
        if np.any(positions[:, 0] < radii - 1e-8) or np.any(positions[:, 0] > w - radii + 1e-8):
            valid = False
        if np.any(positions[:, 1] < radii - 1e-8) or np.any(positions[:, 1] > h - radii + 1e-8):
            valid = False
        if valid:
             dx_c = positions[j_idx, 0] - positions[i_idx, 0]
             dy_c = positions[j_idx, 1] - positions[i_idx, 1]
             dist_c = np.sqrt(dx_c * dx_c + dy_c * dy_c)
             min_dist_c = radii[i_idx] + radii[j_idx]
             if np.any(dist_c < min_dist_c - 1e-8):
                 valid = False
        
        if valid:
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii]).copy()
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
