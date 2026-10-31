import numpy as np
import time
import random

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute pair indices for vectorized overlap checks
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # State
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # 1. Initialization: Hexagonal-like grid to start with large radii
    # We want to start with a configuration that is "tight" but valid.
    # A grid of 5x5=25 + 1 center? Or just 5x5 grid.
    # Let's use a jittered grid.
    side = 5
    indices = 0
    for i in range(side):
        for j in range(side):
            if indices < N:
                x[indices] = (i + 0.5) / side
                y[indices] = (j + 0.5) / side
                r[indices] = 0.5 / side * 0.95 # Slightly smaller than grid cell to allow initial validity
                indices += 1
    
    # Add some jitter to break symmetry
    x += np.random.uniform(-0.01, 0.01, N)
    y += np.random.uniform(-0.01, 0.01, N)
    
    # Ensure initial validity (should be mostly valid, but let's check and shrink if needed)
    for _ in range(100):
        valid = True
        # Check boundaries
        if np.any(x < r) or np.any(1-x < r) or np.any(y < r) or np.any(1-y < r):
            valid = False
            # Shrink all
            r *= 0.9
            continue
            
        # Check pairs
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[i_idx] + r[j_idx] > dist):
            valid = False
            r *= 0.9
            continue
            
        if valid:
            break
            
    best_score = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    # 2. Optimization Loop: "Grow and Push"
    # Strategy:
    # Pick a random circle i.
    # Try to increase r_i.
    # If it hits a wall or another circle, "push" the obstruction away.
    # The push is a hard constraint resolution: move the colliding objects apart.
    
    # We will use a "simulated annealing" style temperature to control how aggressive the pushes are
    # and whether we accept temporary overlaps (which we resolve immediately) or just stop growing.
    
    # Actually, a simpler and more robust "hard" approach:
    # 1. Pick i.
    # 2. Calculate max possible r_i given current positions and other radii.
    # 3. If r_i < max_r, increase r_i.
    # 4. If r_i == max_r (blocked), apply a "kick" to the blocking object(s) to free up space.
    
    # To make this efficient, we'll iterate many times.
    
    max_iters = 20000
    current_time = start_time
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random circle to grow
        i = np.random.randint(N)
        
        # Determine the maximum radius i can have without violating constraints
        # Constraints:
        # 1. x_i - r_i >= 0  => r_i <= x_i
        # 2. 1 - x_i - r_i >= 0 => r_i <= 1 - x_i
        # 3. y_i - r_i >= 0 => r_i <= y_i
        # 4. 1 - y_i - r_i >= 0 => r_i <= 1 - y_i
        max_r = min(x[i], 1-x[i], y[i], 1-y[i])
        
        # 5. Pairwise: dist(i, j) >= r_i + r_j => r_i <= dist(i, j) - r_j
        # We need to check all j != i
        # Vectorized check
        dx = x - x[i]
        dy = y - y[i]
        dist = np.sqrt(dx*dx + dy*dy)
        # dist[i] is 0, so r_i <= -r_i (impossible if r_i>0). We exclude i.
        # For j != i: r_i <= dist[i, j] - r[j]
        
        # We only care about j where dist[i, j] - r[j] < current_max_r
        # Let's compute the limit for all j
        limits = dist - r
        limits[i] = np.inf # Ignore self
        
        # The max radius is the min of the boundary limits and the pairwise limits
        max_r = min(max_r, np.min(limits))
        
        # Ensure non-negative
        max_r = max(max_r, 0.0)
        
        # If we can grow, grow by a small step or to the max?
        # Growing to the max immediately creates a "tight" constraint.
        # Let's grow by a fraction of the gap to allow gradual adjustment.
        # Or, grow to max, then push.
        
        if max_r > r[i] + 1e-6:
            # Grow
            # Let's grow to 90% of the max to keep some slack, or just to max?
            # Growing to max is better for packing density.
            r[i] = max_r
        else:
            # We are blocked. We need to push something.
            # Identify the blocking constraint.
            # Check boundaries
            block_x_left = (x[i] - r[i] < 1e-6)
            block_x_right = (1 - x[i] - r[i] < 1e-6)
            block_y_bot = (y[i] - r[i] < 1e-6)
            block_y_top = (1 - y[i] - r[i] < 1e-6)
            
            # Check pairwise
            # Find j that is blocking
            # dist(i, j) - r[j] <= r[i] + eps
            # dist(i, j) <= r[i] + r[j] + eps
            blocking_js = np.where(dist <= r[i] + r + 1e-6)[0]
            blocking_js = blocking_js[blocking_js != i]
            
            if len(blocking_js) == 0 and not (block_x_left or block_x_right or block_y_bot or block_y_top):
                # Numerical error? Try to nudge i slightly
                x[i] += np.random.uniform(-0.001, 0.001)
                y[i] += np.random.uniform(-0.001, 0.001)
                continue
            
            # Push Strategy:
            # If blocked by a neighbor j, push j away from i.
            # If blocked by wall, push i away from wall? No, i is fixed in this step.
            # Actually, we can move i as well.
            
            # Let's pick a random blocking entity to push.
            # If there are neighbors, prefer pushing a neighbor.
            if len(blocking_js) > 0:
                # Pick a random blocking neighbor
                j = blocking_js[np.random.randint(len(blocking_js))]
                
                # Calculate push direction
                dx_ij = x[j] - x[i]
                dy_ij = y[j] - y[i]
                dist_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                
                if dist_ij < 1e-9:
                    # Overlapping centers?
                    dx_ij = np.random.uniform(-1, 1)
                    dy_ij = np.random.uniform(-1, 1)
                    dist_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                
                # We want to increase dist_ij to r[i] + r[j]
                # Current dist is dist_ij.
                # Required dist is r[i] + r[j].
                # Gap = (r[i] + r[j]) - dist_ij
                # We need to move j by Gap/2 in direction (dx_ij, dy_ij)/dist_ij
                # And move i by Gap/2 in opposite direction.
                
                gap = (r[i] + r[j]) - dist_ij
                # Add a small extra push to actually create space for growth
                push_mag = gap * 0.5 + 0.005
                
                nx = dx_ij / dist_ij
                ny = dy_ij / dist_ij
                
                # Update positions
                x[j] += nx * push_mag
                y[j] += ny * push_mag
                x[i] -= nx * push_mag
                y[i] -= ny * push_mag
                
            else:
                # Blocked by wall
                # Move i away from wall
                move_mag = 0.005
                if block_x_left:
                    x[i] += move_mag
                if block_x_right:
                    x[i] -= move_mag
                if block_y_bot:
                    y[i] += move_mag
                if block_y_top:
                    y[i] -= move_mag
                
        # After moving, we must ensure validity (no overlaps introduced by the push)
        # The push might cause j to overlap with k, or i to overlap with wall.
        # We need a "Relaxation" step to fix any new overlaps.
        
        # Simple relaxation: Iterate a few times to push apart any overlaps.
        # This is a local "Molecular Dynamics" step but with hard constraints.
        for _ in range(10):
            # Check all pairs
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            dist = np.maximum(dist, 1e-9)
            
            overlap = r[i_idx] + r[j_idx] - dist
            mask = overlap > 1e-6
            
            if not np.any(mask):
                break
                
            # Resolve overlaps by moving centers apart
            # Move each circle by half the overlap along the connecting line
            ii = i_idx[mask]
            jj = j_idx[mask]
            ox = dx[mask] / dist[mask]
            oy = dy[mask] / dist[mask]
            
            # Update x, y
            # We need to accumulate displacements
            disp_x = np.zeros(N)
            disp_y = np.zeros(N)
            
            disp_x[ii] += ox * overlap[mask] * 0.5
            disp_y[ii] += oy * overlap[mask] * 0.5
            disp_x[jj] -= ox * overlap[mask] * 0.5
            disp_y[jj] -= oy * overlap[mask] * 0.5
            
            x += disp_x
            y += disp_y
            
        # Check boundaries and clip/shrink
        # If x < r, push x right? Or shrink r?
        # Shrinking r is safer for validity. Pushing x might cause new overlaps.
        # Let's clip positions to [r, 1-r] and shrink r if necessary.
        
        # Boundary handling:
        # For each circle, ensure x in [r, 1-r] and y in [r, 1-r]
        # If x < r, set x = r. If x > 1-r, set x = 1-r.
        # This might create overlaps. The relaxation loop above handles overlaps.
        # But moving x to boundary might push into another circle.
        # So we do:
        # 1. Clip x, y to valid range.
        # 2. Run relaxation again.
        
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        # Final check and shrink if still invalid (rare if relaxation is good)
        # We'll do a strict validity check and shrink globally if needed
        # This is expensive, so we do it every 100 iterations
        if it % 100 == 0:
             valid = True
             # Check boundaries (should be ok due to clip)
             # Check pairs
             dx = x[i_idx] - x[j_idx]
             dy = y[i_idx] - y[j_idx]
             dist = np.sqrt(dx*dx + dy*dy)
             if np.any(r[i_idx] + r[j_idx] > dist + 1e-9):
                 # Shrink all radii slightly
                 r *= 0.95
                 # Re-clip
                 x = np.clip(x, r, 1-r)
                 y = np.clip(y, r, 1-r)
        
        # Track best
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
            
    # Final validity check and projection
    bx, by, br = best_state
    
    # Ensure strict validity for the final output
    # 1. Boundary
    br = np.minimum(br, np.minimum(bx, 1-bx))
    br = np.minimum(br, np.minimum(by, 1-by))
    
    # 2. Pairwise
    # If any overlap, shrink radii iteratively
    for _ in range(100):
        dx = bx[i_idx] - bx[j_idx]
        dy = by[i_idx] - by[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        violations = br[i_idx] + br[j_idx] - dist
        if np.max(violations) < 1e-9:
            break
        # Shrink the larger radius? Or both?
        # Shrink both proportionally to their radius?
        # Simple: shrink all by a factor
        br *= 0.98
        br = np.minimum(br, np.minimum(bx, 1-bx))
        br = np.minimum(br, np.minimum(by, 1-by))
        
    final_state = np.column_stack([bx, by, br])
    return final_state

# EVOLVE-BLOCK-END
