"""Physics-based circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N circles in a rectangle of perimeter 4 (i.e., width + height = 2).
    Maximize sum of radii.
    
    Approach: Repulsion-based local search.
    1. Try multiple aspect ratios.
    2. For each aspect ratio, initialize a grid of circles.
    3. Iteratively increase radii and resolve overlaps by pushing circles apart.
    4. Return the configuration with the maximum sum of radii that is valid.
    """
    best_result = None
    best_score = -np.inf
    
    # Try a range of aspect ratios. 
    # Symmetry: aspect and 1/aspect are equivalent (just swap w,h).
    # We can restrict to aspect >= 1.0 or just try a wide range.
    # For N=21, square (1.0) is a good guess, but 1.2 or 0.8 might be better.
    aspects = np.linspace(0.5, 2.0, 15)
    
    for aspect in aspects:
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Initialize positions in a grid
        # Use a spiral or grid. Grid is simpler.
        # We want to place 21 circles. 
        # 5x5 grid = 25 slots. We use 21.
        # Or 4x6 = 24 slots.
        # Let's use a generic grid that fits N.
        cols = int(np.ceil(np.sqrt(N * (w/h)))) # Approx cols based on aspect
        rows = int(np.ceil(N / cols))
        
        # Refine cols/rows to be closer to aspect ratio
        # Actually, just use a standard grid that covers the area.
        # Let's try to fit them in a grid that matches the aspect ratio roughly.
        # If w > h, more cols than rows.
        
        # Simple heuristic initialization:
        # Place circles in a grid that is as square as possible in terms of cell count,
        # but scaled to w and h.
        
        # Let's just use a 5x5 grid and drop the last 4, or similar.
        # Better: Use a "packed" initialization.
        # For N=21, a 5x5 grid with 4 corners removed? No, just 5x5 grid, take first 21.
        
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            # Center of each cell
            positions[i, 0] = (col + 0.5) * w / side
            positions[i, 1] = (row + 0.5) * h / side
            
        # Initial radii
        # Max possible radius is roughly min(w, h) / side
        # Let's start small and grow.
        r_current = 0.01
        
        # Relaxation parameters
        max_grow_steps = 500
        grow_factor = 1.001
        relax_steps = 50
        tolerance = 1e-6
        
        result = _relax_packing(positions, r_current, w, h, N, max_grow_steps, grow_factor, relax_steps)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
                
    if best_result is None:
        # Fallback: very small circles
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            result[i, 0] = (col + 0.5) * w / side
            result[i, 1] = (row + 0.5) * h / side
            result[i, 2] = 0.01
        return result
        
    return best_result


def _relax_packing(positions, r_init, w, h, N, max_grow_steps, grow_factor, relax_steps):
    """
    Relaxation algorithm to maximize radii.
    """
    positions = positions.copy()
    radii = np.full(N, r_init)
    
    for step in range(max_grow_steps):
        # Grow radii
        radii *= grow_factor
        
        # Check if any radius exceeds max possible (min(w,h)/2 is not the limit, but w/2 and h/2 are)
        # Actually, max r is limited by the smallest dimension if centered, but generally
        # a circle can have r up to min(w, h)/2 if it's the only one.
        # With packing, max r is smaller.
        # If r_init * grow_factor^step > 0.5, we are definitely too big.
        if radii[0] > 0.5:
            break
            
        # Relax overlaps and boundaries
        valid = True
        for rel_step in range(relax_steps):
            # Resolve overlaps
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist:
                        # Push apart
                        # Handle zero distance
                        if dist < 1e-10:
                            dx = 1e-5
                            dy = 0.0
                            dist = 1e-5
                            
                        # Overlap amount
                        overlap = min_dist - dist
                        # Push each circle by half the overlap
                        push_x = (dx / dist) * (overlap / 2.0)
                        push_y = (dy / dist) * (overlap / 2.0)
                        
                        positions[i, 0] -= push_x
                        positions[i, 1] -= push_y
                        positions[j, 0] += push_x
                        positions[j, 1] += push_y
                        moved = True
                        
            # Resolve boundaries
            for i in range(N):
                r = radii[i]
                x, y = positions[i, 0], positions[i, 1]
                
                if x < r:
                    positions[i, 0] = r
                    moved = True
                elif x > w - r:
                    positions[i, 0] = w - r
                    moved = True
                    
                if y < r:
                    positions[i, 1] = r
                    moved = True
                elif y > h - r:
                    positions[i, 1] = h - r
                    moved = True
            
            # If no movement, we are stable for this radius
            if not moved:
                break
                
        # Final validation
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        # Check boundaries
        if valid:
            for i in range(N):
                r = radii[i]
                x, y = positions[i, 0], positions[i, 1]
                if x < r - 1e-5 or x > w - r + 1e-5 or y < r - 1e-5 or y > h - r + 1e-5:
                    valid = False
                    break
                    
        if not valid:
            # Backtrack: reduce radii to previous valid state?
            # For simplicity, if it becomes invalid, we stop growing.
            # The previous iteration was valid (or we assume it was).
            # We need to return the last valid configuration.
            # So we should keep a copy of the last valid state.
            
            # Let's restructure: keep last_valid
            pass # Logic handled below
            
    # The loop above doesn't explicitly store last valid. 
    # Let's rewrite the loop to store the best valid state found.
    
    # Actually, the current loop modifies `positions` and `radii` in place.
    # If it becomes invalid, the state is corrupted.
    # We need to check validity BEFORE accepting the growth.
    
    # Let's restart with a cleaner implementation.
    return _relax_packing_robust(positions, r_init, w, h, N, max_grow_steps, grow_factor, relax_steps)


def _relax_packing_robust(positions_init, r_init, w, h, N, max_grow_steps, grow_factor, relax_steps):
    positions = positions_init.copy()
    radii = np.full(N, r_init)
    
    last_valid_positions = positions.copy()
    last_valid_radii = radii.copy()
    
    for step in range(max_grow_steps):
        # Try to grow
        new_radii = radii * grow_factor
        
        # If radii are too large to fit in the box even alone, stop
        if np.any(new_radii > w/2) or np.any(new_radii > h/2):
            break
            
        # Attempt to relax with new_radii
        # We need to relax positions to fit new_radii
        # Start from current positions
        test_positions = positions.copy()
        test_radii = new_radii.copy()
        
        # Relaxation loop
        converged = False
        for rel_step in range(relax_steps):
            moved = False
            # Resolve overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = test_positions[j, 0] - test_positions[i, 0]
                    dy = test_positions[j, 1] - test_positions[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = test_radii[i] + test_radii[j]
                    
                    if dist < min_dist:
                        if dist < 1e-10:
                            dx = 1e-5
                            dy = 0.0
                            dist = 1e-5
                            
                        overlap = min_dist - dist
                        push_x = (dx / dist) * (overlap / 2.0)
                        push_y = (dy / dist) * (overlap / 2.0)
                        
                        test_positions[i, 0] -= push_x
                        test_positions[i, 1] -= push_y
                        test_positions[j, 0] += push_x
                        test_positions[j, 1] += push_y
                        moved = True
                        
            # Resolve boundaries
            for i in range(N):
                r = test_radii[i]
                x, y = test_positions[i, 0], test_positions[i, 1]
                
                if x < r:
                    test_positions[i, 0] = r
                    moved = True
                elif x > w - r:
                    test_positions[i, 0] = w - r
                    moved = True
                    
                if y < r:
                    test_positions[i, 1] = r
                    moved = True
                elif y > h - r:
                    test_positions[i, 1] = h - r
                    moved = True
            
            if not moved:
                converged = True
                break
                
        # Check if the relaxed configuration is valid
        if converged:
            # Strict check
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = test_positions[j, 0] - test_positions[i, 0]
                    dy = test_positions[j, 1] - test_positions[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < test_radii[i] + test_radii[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for i in range(N):
                    r = test_radii[i]
                    x, y = test_positions[i, 0], test_positions[i, 1]
                    if x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
                        valid = False
                        break
                        
            if valid:
                # Accept this step
                positions = test_positions
                radii = test_radii
                last_valid_positions = positions.copy()
                last_valid_radii = radii.copy()
            else:
                # Failed to relax to valid state, stop
                break
        else:
            # Did not converge, likely stuck, stop
            break
            
    if np.sum(last_valid_radii) <= 0:
        return None
        
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = last_valid_positions[i, 0]
        result[i, 1] = last_valid_positions[i, 1]
        result[i, 2] = last_valid_radii[i]
        
    return result

# EVOLVE-BLOCK-END
