"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """
    circles: (N, 3) array of [x, y, r]
    Returns (score, is_valid)
    score = sum of radii, but only if all constraints are satisfied.
    Constraints:
    1. All circles inside bounding rectangle.
    2. No overlapping circles.
    3. Bounding rectangle perimeter <= 4.
    """
    if len(circles) != N:
        return 0.0, False
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check radii positive
    if np.any(r <= 0):
        return 0.0, False
    
    # Check bounds: find min/max
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w < 1e-9 or h < 1e-9:
        return 0.0, False
        
    perimeter = 2 * (w + h)
    if perimeter > 4.0 + 1e-9:
        return 0.0, False
        
    # Check overlaps
    # For each pair, distance between centers >= sum of radii
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
                
    return np.sum(r), True

def _scale_to_fit(circles, target_perim=4.0):
    """
    Scale and translate circles to fit in a rectangle of perimeter target_perim.
    Returns new circles array.
    """
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    min_x = np.min(x - r)
    max_x = np.max(x + r)
    min_y = np.min(y - r)
    max_y = np.max(y + r)
    
    w = max_x - min_x
    h = max_y - min_y
    
    if w == 0 or h == 0:
        return circles
        
    current_perim = 2 * (w + h)
    
    # We want to scale such that 2*(w*s + h*s) <= target_perim
    # s <= target_perim / current_perim
    s = target_perim / current_perim
    
    # Apply scaling relative to bottom-left corner of bounding box
    new_x = (x - min_x) * s
    new_y = (y - min_y) * s
    new_r = r * s
    
    return np.column_stack((new_x, new_y, new_r))

def _pbd_solve(x, y, r, w, h, max_iters=1000, tol=1e-6):
    """
    Position Based Dynamics solver.
    x, y, r: arrays of length N.
    w, h: rectangle dimensions.
    Returns optimized x, y, r.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # We fix r to its initial value? Or allow it to change?
    # The task is to maximize sum of radii.
    # In PBD, we can treat radius as a variable that we want to maximize.
    # However, standard PBD fixes properties.
    # Strategy:
    # 1. Start with some configuration.
    # 2. Use PBD to resolve overlaps and wall collisions while trying to expand radii?
    # Actually, a simpler approach for "maximize sum of radii" with fixed aspect ratio:
    # We can fix the rectangle size (w, h) and try to find the maximal packing.
    # If we fix w and h, the problem is to place circles to maximize sum(r).
    # This is equivalent to finding the maximal non-overlapping circles in a fixed box.
    
    # Let's use a different approach:
    # 1. Generate a random initial configuration of N circles in a unit square.
    # 2. Assign initial radii.
    # 3. Use PBD to resolve overlaps (push circles apart) and wall collisions (push inside).
    # 4. In each iteration, also try to increase radii slightly?
    # No, PBD is for constraints.
    
    # Better approach:
    # We want to maximize sum(r).
    # Let's fix the rectangle to be w x h (perimeter 4).
    # We want to place circles.
    # We can use a gradient-free method or a specific PBD variant.
    
    # Let's try this:
    # 1. Initialize circles with small radii, random positions in [0, w] x [0, h].
    # 2. Iteratively:
    #    a. Increase all radii by a small amount dr.
    #    b. Run PBD relaxation to resolve overlaps and wall constraints.
    #    c. If relaxation fails (circles can't fit), revert radii or adjust.
    
    # This is essentially "greedy expansion with relaxation".
    
    # Let's implement the relaxation step (PBD) first.
    # PBD Step:
    # For each pair of circles, if they overlap, push them apart along the center line.
    # For each circle and wall, if it overlaps, push it inside.
    
    def relax(x, y, r, w, h, iters=100):
        for _ in range(iters):
            # Wall constraints
            for i in range(N):
                # Left
                if x[i] - r[i] < 0:
                    x[i] = r[i]
                # Right
                if x[i] + r[i] > w:
                    x[i] = w - r[i]
                # Bottom
                if y[i] - r[i] < 0:
                    y[i] = r[i]
                # Top
                if y[i] + r[i] > h:
                    y[i] = h - r[i]
            
            # Pair constraints
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist:
                        if dist < 1e-6:
                            # If they are at the same spot, push randomly
                            dx = np.random.uniform(-1, 1)
                            dy = np.random.uniform(-1, 1)
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < 1e-6:
                                dx, dy = 1.0, 0.0
                                dist = 1.0
                            
                        overlap = min_dist - dist
                        # Push apart by half the overlap each
                        # Weight by inverse mass? Assume equal mass.
                        push_x = (dx / dist) * (overlap / 2.0)
                        push_y = (dy / dist) * (overlap / 2.0)
                        
                        x[i] -= push_x
                        y[i] -= push_y
                        x[j] += push_x
                        y[j] += push_y
                        
                        # Re-check walls after push? 
                        # PBD usually does multiple passes.
                        # For simplicity, we'll just do one pass of walls at the start of the next iter.
                        # But this might push out of walls.
                        # Let's re-apply wall constraints immediately for these two?
                        # Or just rely on the next iteration.
                        # Given the strictness, let's re-apply walls for i and j.
                        
                        # Wall i
                        if x[i] - r[i] < 0: x[i] = r[i]
                        if x[i] + r[i] > w: x[i] = w - r[i]
                        if y[i] - r[i] < 0: y[i] = r[i]
                        if y[i] + r[i] > h: y[i] = h - r[i]
                        
                        # Wall j
                        if x[j] - r[j] < 0: x[j] = r[j]
                        if x[j] + r[j] > w: x[j] = w - r[j]
                        if y[j] - r[j] < 0: y[j] = r[j]
                        if y[j] + r[j] > h: y[j] = h - r[j]
                        
        return x, y

    # Main loop
    best_sum_r = np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Start with small radii?
    # The input r might be small. Let's try to expand.
    
    # If initial r is already large, we just relax.
    # If initial r is small, we can try to grow them.
    
    # Strategy:
    # 1. Relax the current configuration.
    # 2. Try to increase all radii by a factor of 1.01.
    # 3. Relax again.
    # 4. If valid, keep. If not, reduce factor.
    
    # But how do we know if it's "valid"?
    # The relax function forces validity (approximately).
    # If the circles are too big for the box, relax will fail to separate them fully,
    # leading to residual overlaps.
    
    # Let's check for residual overlaps after relaxation.
    
    def check_overlap(x, y, r, w, h):
        # Walls
        for i in range(N):
            if x[i] - r[i] < -1e-5 or x[i] + r[i] > w + 1e-5 or y[i] - r[i] < -1e-5 or y[i] + r[i] > h + 1e-5:
                return False
        # Pairs
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-5:
                    return False
        return True

    # Relax initial state
    x, y = relax(x, y, r, w, h, iters=500)
    
    if check_overlap(x, y, r, w, h):
        best_sum_r = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Try to grow
    growth_factor = 1.05
    for step in range(50):
        if time.time() > DEADLINE:
            break
            
        new_r = r * growth_factor
        # Relax with new radii
        new_x, new_y = relax(x, y, new_r, w, h, iters=200)
        
        if check_overlap(new_x, new_y, new_r, w, h):
            # Success
            x, y, r = new_x, new_y, new_r
            best_sum_r = np.sum(r)
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            # Maybe increase growth factor? Or keep it constant?
            # Let's keep it constant for stability, or slightly increase.
            growth_factor = 1.02
        else:
            # Failure
            # Try smaller growth
            growth_factor = max(1.01, growth_factor * 0.8)
            if growth_factor <= 1.001:
                break
            # Try relaxing with current r again to see if we can squeeze?
            # No, if growth failed, we stick to current best.
            # We can try a different random restart or perturbation.
            # For now, break or try smaller.
            # Let's try smaller.
            continue
            
    return best_x, best_y, best_r

def construct():
    start_time = time.time()
    best_score = 0.0
    best_circles = None
    
    # Try different aspect ratios
    # Perimeter 2(w+h) = 4 => w+h = 2.
    # w in (0, 2). h = 2 - w.
    # Aspect ratio w/h.
    # Let's try w from 0.5 to 1.5 (since symmetric, w/h and h/w are similar).
    # Actually, just iterate w.
    
    # Also try different initial configurations.
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    # Use a grid of side x side
    # But N=21, side=5. 5x5=25. We take 21.
    
    for trial in range(50): # Max trials
        if time.time() > DEADLINE:
            break
            
        # Random aspect ratio
        # w + h = 2
        # w in [0.5, 1.5]
        w = np.random.uniform(0.5, 1.5)
        h = 2.0 - w
        
        # Initialize positions
        # Grid based
        # 5x5 grid
        cols = 5
        rows = 5
        # We need 21 circles.
        # Let's place them in a 5x5 grid, skipping some corners to make it more compact?
        # Or just random jitter.
        
        # Better: Random positions
        x = np.random.uniform(0, w, N)
        y = np.random.uniform(0, h, N)
        
        # Initial radii: small
        # Max possible radius is min(w, h) / 2 roughly?
        # Let's start with small radii, e.g., 0.05
        r = np.full(N, 0.05)
        
        # Run PBD solver
        opt_x, opt_y, opt_r = _pbd_solve(x, y, r, w, h)
        
        # Scale to fit perimeter 4?
        # The PBD solver kept them within w x h.
        # So perimeter is 2(w+h) = 4.
        # We just need to return them.
        
        circles = np.column_stack((opt_x, opt_y, opt_r))
        
        # Validate
        score, valid = _validate_and_score(circles)
        
        if valid and score > best_score:
            best_score = score
            best_circles = circles.copy()
            
            # Early stop if we hit a very high score?
            # Theoretical max?
            # For 21 circles, sum of radii...
            # If we pack them tightly, average radius might be around 0.1-0.15?
            # Let's say if score > 2.0, it's good.
            
        # If time is running out, break
        if time.time() > DEADLINE - 5:
            break
            
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                # Square box 1x1
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append((x, y, r))
                count += 1
        best_circles = np.array(cells)
        # Scale to perimeter 4?
        # 1x1 box has perimeter 4. So it's already valid.
        # Check if it fits 21 circles.
        # 5x5 grid fits 25. We take 21.
        # Radii: 0.5/5 = 0.1. Minus 0.01 = 0.09.
        # Sum = 21 * 0.09 = 1.89.
        
    return best_circles

# EVOLVE-BLOCK-END
