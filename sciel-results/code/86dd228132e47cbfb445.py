import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Initial guess: Hexagonal packing in a square
    # Try a few different initial configurations and aspect ratios
    best_result = None
    best_score = -np.inf
    
    # Define a function to evaluate a configuration
    def evaluate(circles, w, h):
        # circles: (N, 3) array [x, y, r]
        # Check validity
        if np.any(circles[:, 2] <= 0):
            return -np.inf
        # Boundary check
        if np.any(circles[:, 0] - circles[:, 2] < -1e-9) or \
           np.any(circles[:, 0] + circles[:, 2] > w + 1e-9) or \
           np.any(circles[:, 1] - circles[:, 2] < -1e-9) or \
           np.any(circles[:, 1] + circles[:, 2] > h + 1e-9):
            return -np.inf
        # Pairwise distance check
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                if dist < circles[i, 2] + circles[j, 2] - 1e-9:
                    return -np.inf
        return np.sum(circles[:, 2])
    
    def optimize(circles_init, w_init, h_init, max_iters=5000):
        circles = circles_init.copy()
        w = w_init
        h = h_init
        # We can also optimize w, but let's fix aspect ratio first for speed, 
        # or try a few fixed aspect ratios.
        # Actually, optimizing w is hard with simple gradient. 
        # Let's stick to square (w=1, h=1) and maybe a few rectangular shapes.
        
        # Repulsion/Growth algorithm
        # 1. Start with small radii
        # 2. Grow radii, push apart on collision
        
        # Scale down radii to ensure no initial overlap
        # Find max possible radius roughly
        # Just start with small r
        r_scale = 0.5
        circles[:, 2] *= r_scale
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Resolve overlaps
            for _ in range(5): # Multiple relaxation steps per growth step
                for i in range(N):
                    # Boundary push
                    if circles[i, 0] < circles[i, 2]:
                        circles[i, 0] = circles[i, 2]
                    elif circles[i, 0] > w - circles[i, 2]:
                        circles[i, 0] = w - circles[i, 2]
                        
                    if circles[i, 1] < circles[i, 2]:
                        circles[i, 1] = circles[i, 2]
                    elif circles[i, 1] > h - circles[i, 2]:
                        circles[i, 1] = h - circles[i, 2]
                        
                # Pairwise push
                for i in range(N):
                    for j in range(i+1, N):
                        dx = circles[i, 0] - circles[j, 0]
                        dy = circles[i, 1] - circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = circles[i, 2] + circles[j, 2]
                        if dist < min_dist:
                            if dist < 1e-12:
                                dx = 1e-12
                                dy = 0
                                dist = 1e-12
                            # Push apart
                            overlap = min_dist - dist
                            # Move each by half the overlap
                            push_x = (dx / dist) * overlap * 0.5
                            push_y = (dy / dist) * overlap * 0.5
                            
                            circles[i, 0] += push_x
                            circles[i, 1] += push_y
                            circles[j, 0] -= push_x
                            circles[j, 1] -= push_y
                            
                            # Re-check boundaries after push
                            # This can be tricky, simple clamp might cause oscillation
                            # but usually works well enough for packing
                            
            # 2. Grow radii
            # Estimate how much we can grow
            # Just grow by a small fixed amount, or adaptive
            growth = 0.001
            # Adaptive: find min clearance
            min_clearance = np.inf
            # Check boundaries
            for i in range(N):
                min_clearance = min(min_clearance, circles[i, 0], w - circles[i, 0])
                min_clearance = min(min_clearance, circles[i, 1], h - circles[i, 1])
            
            # Check pairs
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    clearance = dist - (circles[i, 2] + circles[j, 2])
                    min_clearance = min(min_clearance, clearance)
            
            if min_clearance <= 0:
                # Should not happen if relaxation worked, but safety
                growth = 0
            else:
                # Grow by a fraction of min clearance
                growth = min(0.001, min_clearance * 0.5)
                
            circles[:, 2] += growth
            
            # If growth is 0 for a long time, we might be stuck, but let's keep trying
            # or break if very small
            if growth < 1e-6:
                # Try a small random perturbation to escape local minima?
                # Or just break
                pass

        return circles, w, h

    # Generate initial configurations
    configs = []
    
    # 1. Grid 5x5 (21 circles)
    # 5 cols, 4 rows + 1? No, 5x5 grid has 25. Take 21.
    # Better: 5 cols, 5 rows, but remove 4 corners? Or just first 21 in row-major?
    # Row-major 5x5:
    # (0,0) (0,1) ... (0,4)
    # (1,0) ...
    # Let's use a hexagonal pattern for better density.
    
    # Hexagonal packing in 1x1 square
    # Rows of circles.
    # r = 0.1 approx.
    # y = r, 3r, 5r...
    # x offset by r for odd rows.
    
    def make_hex_config(w, h):
        # Estimate r
        r = 0.1
        circles = []
        y = r
        row = 0
        while y + r <= h:
            x_offset = 0 if row % 2 == 0 else r
            x = r + x_offset
            while x + r <= w:
                circles.append([x, y, r])
                x += 2*r
            y += np.sqrt(3)*r # Vertical distance in hex packing
            row += 1
            if len(circles) >= N:
                break
        if len(circles) < N:
            # Fill rest with small circles or just return what we have?
            # We need exactly N.
            # If we don't have enough, add small ones in remaining space?
            # Or just use grid fallback.
            return None
        return np.array(circles[:N])
    
    def make_grid_config(w, h, cols):
        rows = int(np.ceil(N / cols))
        circles = []
        # Distribute N circles in cols x rows grid
        # Center them in the rectangle
        # Simple: divide w into cols, h into rows
        # But we need to fit exactly N.
        # If N is not cols*rows, some cells are empty.
        # To maximize sum of radii, we should probably use a rectangle that fits the grid tightly.
        # But w+h=2 is fixed.
        # Let's just place them in a grid within w x h.
        
        # Calculate cell size
        cw = w / cols
        ch = h / rows
        
        # Place circles in the first N cells
        count = 0
        for r_idx in range(rows):
            for c_idx in range(cols):
                if count >= N:
                    break
                cx = (c_idx + 0.5) * cw
                cy = (r_idx + 0.5) * ch
                # Radius is half the min cell dimension, minus small epsilon
                # But they might overlap if cw != ch
                # To be safe, use min(cw, ch)/2
                rad = 0.5 * min(cw, ch) * 0.95
                circles.append([cx, cy, rad])
                count += 1
        return np.array(circles)
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7]
    
    for ar in aspect_ratios:
        # w + h = 2, w/h = ar => h = w/ar => w(1 + 1/ar) = 2 => w = 2 / (1 + 1/ar)
        w = 2.0 / (1.0 + 1.0/ar)
        h = 2.0 - w
        
        # Try Grid
        for cols in [4, 5, 6]:
            init = make_grid_config(w, h, cols)
            if init is not None and len(init) == N:
                # Optimize
                res, _, _ = optimize(init, w, h)
                score = evaluate(res, w, h)
                if score > best_score:
                    best_score = score
                    best_result = res.copy()
                    
        # Try Hex
        init = make_hex_config(w, h)
        if init is not None and len(init) == N:
            res, _, _ = optimize(init, w, h)
            score = evaluate(res, w, h)
            if score > best_score:
                best_score = score
                best_result = res.copy()
        
        if time.time() > deadline:
            break

    if best_result is None:
        # Fallback to simple grid
        init = make_grid_config(1.0, 1.0, 5)
        if init is None:
             # Should not happen
             return np.zeros((N, 3))
        res, _, _ = optimize(init, 1.0, 1.0)
        best_result = res

    # Final sanity check and cleanup
    # Ensure all radii are positive and circles are within bounds
    # The optimizer should handle this, but let's clamp just in case
    # Note: Clamping might cause overlaps if done naively.
    # But the optimizer already did relaxation.
    
    return best_result
