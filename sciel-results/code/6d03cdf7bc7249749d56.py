"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_solution = None
    best_value = -np.inf
    
    # We will optimize the aspect ratio 'a' where w = 2a/(1+a), h = 2/(1+a)
    # Perimeter 2(w+h) = 4 => w+h=2.
    # Let's parameterize by w directly? Or aspect ratio.
    # Let's try a range of aspect ratios and run force-directed relaxation on each.
    
    # Force Directed Relaxation Strategy:
    # 1. Initialize circles in a grid.
    # 2. Iteratively:
    #    a. Calculate repulsive forces between overlapping circles (and boundary violations).
    #    b. Calculate attractive forces? No, we want to maximize radii.
    #       Actually, standard circle packing often fixes radii and minimizes area, or fixes area and maximizes radii.
    #       Here, we want to maximize sum(r_i).
    #       
    #       Alternative approach: 
    #       Start with small radii. 
    #       In each step, try to increase radii slightly. 
    #       If overlaps occur, push circles apart.
    #       This is essentially a "grow and relax" algorithm.
    #
    #       Let's implement a vectorized "Grow and Relax" with annealing.
    
    def relax_packing(w, h, n_circles, max_iter=2000, seed=0):
        """
        Vectorized force-directed relaxation to maximize sum of radii.
        """
        rng = np.random.default_rng(seed)
        
        # Initialize positions in a grid
        side = int(np.ceil(np.sqrt(n_circles)))
        cols = side
        rows = int(np.ceil(n_circles / side))
        
        # Actual grid dimensions for initialization
        # We want to fit n_circles into w x h
        # Let's just use a simple grid over the whole rectangle
        xs = np.zeros(n_circles)
        ys = np.zeros(n_circles)
        rs = np.zeros(n_circles)
        
        # Initial grid layout
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Center of cell
                xs[idx] = (j + 0.5) * w / cols
                ys[idx] = (i + 0.5) * h / rows
                # Initial radius: half the min cell size, with some margin
                cell_w = w / cols
                cell_h = h / rows
                r_init = 0.4 * min(cell_w, cell_h)
                rs[idx] = r_init
                idx += 1
        
        # Ensure we have exactly n_circles
        if idx < n_circles:
             # Fill remaining if grid was too small (unlikely with ceil)
             pass
             
        # Optimization Loop
        # We use a "temperature" T to control the strength of radius growth vs. separation.
        # Objective: Maximize sum(r). Constraint: dist(i,j) >= r_i + r_j.
        
        # We can formulate this as minimizing a potential energy:
        # E = -alpha * sum(r) + beta * sum(overlap_violations^2)
        # And minimize E using gradient descent on positions and radii.
        # But radii are coupled to positions.
        
        # Simpler: Iterative "Push and Grow"
        # 1. Increase all radii by a small delta.
        # 2. Resolve overlaps by pushing circles apart.
        # 3. Push circles back into bounds.
        
        # Let's try a different vectorized approach:
        # Use scipy's minimize with a better initial guess and constraints?
        # No, the prompt asks for a DIFFERENT strategy. Force-directed is good.
        
        # Let's implement a vectorized "Repulsion + Growth" step.
        # At each step t:
        # 1. Compute all pairwise distances D_ij.
        # 2. Compute overlaps O_ij = max(0, r_i + r_j - D_ij).
        # 3. Compute forces:
        #    - Repulsive force between i,j: proportional to O_ij * (r_j/(r_i+r_j)) in direction i->j?
        #      Actually, to separate them, we move i away from j and j away from i.
        #      Force on i from j: F_ij = k * O_ij * (unit vector from j to i)
        #    - Boundary forces: Push back if out of bounds.
        # 4. Update positions: x_i += learning_rate * F_i
        # 5. Grow radii: r_i += growth_rate
        #    But if we grow radii, we must ensure we don't explode.
        #    Maybe grow radii only if the "pressure" (total overlap) is low?
        
        # Let's refine the "Grow and Relax":
        # We want to maximize sum(r).
        # Let's use a simulated annealing on the "temperature" of the repulsion.
        
        # Vectorized Repulsion Calculation
        # N=21, so 21x21 matrix is small. 441 elements. Very fast.
        
        current_temp = 1.0
        cooling_rate = 0.999
        growth_step = 0.001
        pos_lr = 0.1
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # 1. Compute pairwise distances and overlaps
            # xs, ys, rs are (N,)
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf) # Avoid self-interaction
            
            min_dist = rs[:, None] + rs[None, :]
            overlap = np.maximum(0, min_dist - dist)
            
            # 2. Compute Forces
            # Force on i due to j is proportional to overlap_ij * (r_j / (r_i + r_j)) ?
            # To be fair, let's just push them apart with force proportional to overlap.
            # The direction is (dx_ij, dy_ij) / dist_ij.
            # Note: dx = x_i - x_j. So vector from j to i is (dx, dy).
            # We want to move i in direction (dx, dy) to increase distance.
            
            # Normalize direction, handle dist=0
            # dist is inf on diagonal, so no division by zero there.
            # For off-diagonal, if dist is very small, add epsilon.
            dist_safe = np.where(dist < 1e-6, 1e-6, dist)
            
            # Force magnitude: overlap * (some factor)
            # We want to distribute the correction.
            # Force on i from j: F = overlap_ij * (dx_ij / dist_ij)
            # Sum over j != i.
            
            fx = np.sum(overlap * (dx / dist_safe), axis=1)
            fy = np.sum(overlap * (dy / dist_safe), axis=1)
            
            # 3. Boundary Forces
            # If x_i - r_i < 0, push right.
            # If x_i + r_i > w, push left.
            # Force is proportional to violation.
            
            left_viol = np.maximum(0, rs - xs)
            right_viol = np.maximum(0, xs + rs - w)
            bottom_viol = np.maximum(0, rs - ys)
            top_viol = np.maximum(0, ys + rs - h)
            
            fx += left_viol - right_viol
            fy += bottom_viol - top_viol
            
            # 4. Update Positions
            # Use a temperature to control the step size?
            # Or just fixed step.
            # Add some noise to escape local minima?
            noise = rng.normal(0, 0.01 * current_temp, size=n_circles)
            xs += pos_lr * fx + noise
            ys += pos_lr * fy + rng.normal(0, 0.01 * current_temp, size=n_circles)
            
            # 5. Grow Radii
            # Only grow if the system is "stable" (low overlap)?
            # Or grow always, and let the repulsion handle it?
            # If we grow too fast, it becomes unstable.
            # Let's grow by a small amount, scaled by temperature?
            # Actually, if we just grow, we will eventually hit a configuration where overlaps are high.
            # The repulsion will push them apart. If they can't be pushed apart (boundary),
            # the overlap will persist.
            # We want to maximize sum(r).
            # So we should always try to grow.
            
            # Let's check total overlap. If it's too high, maybe shrink?
            # No, we want to maximize.
            # Let's just grow.
            rs += growth_step
            
            # 6. Cool down
            current_temp *= cooling_rate
            
            # 7. Check for convergence or timeout
            if current_temp < 1e-3:
                break
                
        # Final verification and cleanup
        # Ensure circles are inside bounds (clip positions)
        xs = np.clip(xs, rs, w - rs)
        ys = np.clip(ys, rs, h - rs)
        
        # One final relaxation step without growth to ensure no overlaps?
        # The problem requires disjoint interiors.
        # If there are overlaps, the solution is invalid (score 0 or penalty).
        # We need to ensure the final state has NO overlaps.
        # If my "grow and relax" leaves overlaps, it's bad.
        
        # Let's do a final "strict" relaxation:
        # Reduce radii slightly until no overlaps?
        # Or just check.
        
        # Check overlaps
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = rs[:, None] + rs[None, :]
        overlap = np.max(min_dist - dist)
        
        if overlap > 1e-6:
            # Solution is invalid. 
            # Strategy: Shrink radii proportionally until valid?
            # Or return a valid sub-solution?
            # Better: Run a "shrink" loop.
            # But we want to maximize.
            # If the force-directed method failed to find a valid packing at this radius,
            # maybe we should have grown slower.
            
            # For now, let's shrink to make it valid.
            # This is a fallback.
            scale = 1.0
            while overlap > 1e-6 and scale > 0.5:
                rs *= scale
                dx = xs[:, None] - xs[None, :]
                dy = ys[:, None] - ys[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(dist, np.inf)
                min_dist = rs[:, None] + rs[None, :]
                overlap = np.max(min_dist - dist)
                scale -= 0.01
                
            if overlap > 1e-6:
                return None, -np.inf # Failed to find valid packing
        
        value = np.sum(rs)
        return np.column_stack((xs, ys, rs)), value

    # Try multiple aspect ratios
    # w + h = 2
    # w = 2 * a / (1+a)
    # h = 2 / (1+a)
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 0.5, 1.3, 1.8, 0.7, 1.1, 1.4]
    
    # Shuffle to explore different geometries
    random.shuffle(aspect_ratios)
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        if w <= 0 or h <= 0:
            continue
            
        # Run relaxation
        # Use different seeds for diversity
        for seed in range(3):
            if time.time() > deadline - 10:
                break
                
            sol, val = relax_packing(w, h, N, seed=seed)
            
            if sol is not None and val > best_value:
                best_value = val
                best_solution = sol
                
    # If no solution found, fallback to simple grid (valid but low score)
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = 0.4 / side
                cells.append([cx, cy, r])
                idx += 1
        best_solution = np.array(cells[:N])
        # Verify validity of fallback
        # This fallback is likely valid if r is small enough.
        
    return best_solution

# EVOLVE-BLOCK-END
