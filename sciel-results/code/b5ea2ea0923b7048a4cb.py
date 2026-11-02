"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    # We want to pack 21 circles in a rectangle with perimeter 4.
    # Let width = w, height = h. 2*(w+h) = 4 => w+h = 2.
    # We can parameterize by aspect ratio r = w/h.
    # Then h = 2/(1+r), w = 2r/(1+r).
    
    # Strategy: 
    # 1. Try a grid-based initial guess for various aspect ratios.
    # 2. Use a simple local optimization (gradient ascent or simulated annealing) to maximize sum of radii.
    # 3. Since we have a hard time limit, we'll do a few restarts with different aspect ratios and initial positions.
    
    best_value = 0.0
    best_result = None
    
    # Number of circles
    n = N
    
    # Try a range of aspect ratios
    # For 21 circles, a roughly square or slightly rectangular arrangement might be good.
    # 21 = 3x7 or 4x5 (with one extra) etc.
    # Let's try aspect ratios from 0.5 to 2.0
    
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
    
    for r in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        h = 2.0 / (1.0 + r)
        w = 2.0 * r / (1.0 + r)
        
        # Initialize positions in a grid
        # Choose grid dimensions close to aspect ratio
        # If r > 1, more columns than rows
        cols = int(np.ceil(np.sqrt(n * r)))
        rows = int(np.ceil(n / cols))
        
        # Adjust to ensure we have at least n cells
        while cols * rows < n:
            if r >= 1:
                cols += 1
            else:
                rows += 1
        
        # Create grid positions
        positions = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Fill the grid
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 0.5) * w / cols
                y = (i + 0.5) * h / rows
                positions[idx] = [x, y]
                # Initial radius: limited by cell size and distance to boundary
                max_r = min(w / cols, h / rows) / 2.0
                radii[idx] = max_r * 0.9  # Start slightly smaller to allow movement
                idx += 1
        
        # Local optimization: maximize sum of radii subject to constraints
        # Constraints:
        # 1. Circles don't overlap: dist(i,j) >= r_i + r_j
        # 2. Circles inside rectangle: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
        
        # We'll use a simple gradient-free approach or a custom local search.
        # Given the non-convexity, let's use a variant of simulated annealing or just random perturbations with acceptance.
        
        # For efficiency, let's do a simple hill climbing with random restarts within this aspect ratio.
        current_value = np.sum(radii)
        best_local_value = current_value
        best_local_positions = positions.copy()
        best_local_radii = radii.copy()
        
        # Number of iterations for local optimization
        max_iters = 5000
        step_size = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random circle to move
            i = np.random.randint(n)
            
            # Try moving it slightly
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dr = np.random.uniform(-step_size/2, step_size/2)
            
            new_x = positions[i, 0] + dx
            new_y = positions[i, 1] + dy
            new_r = radii[i] + dr
            
            # Check if new position is valid
            if new_r < 0.01:  # Minimum radius
                continue
            
            # Check boundary constraints
            if new_x - new_r < 0 or new_x + new_r > w or new_y - new_r < 0 or new_y + new_r > h:
                continue
            
            # Check overlap with other circles
            valid = True
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
                if dist < new_r + radii[j] - 1e-9:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new value
            new_value = current_value + dr
            
            # Accept if better or with some probability (simulated annealing style)
            if new_value > current_value:
                positions[i, 0] = new_x
                positions[i, 1] = new_y
                radii[i] = new_r
                current_value = new_value
                
                if current_value > best_local_value:
                    best_local_value = current_value
                    best_local_positions = positions.copy()
                    best_local_radii = radii.copy()
            else:
                # Accept with small probability to escape local minima
                if np.random.random() < 0.01:
                    positions[i, 0] = new_x
                    positions[i, 1] = new_y
                    radii[i] = new_r
                    current_value = new_value
        
        # Update global best
        if best_local_value > best_value:
            best_value = best_local_value
            best_result = np.column_stack([best_local_positions, best_local_radii])
    
    # If no valid result found, return a default grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
