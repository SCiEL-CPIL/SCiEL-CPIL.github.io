import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_sum = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # For n=21, aspect ratio 1 is a good start, but others might be better.
    # Let's scan aspect ratios from 0.6 to 1.6
    aspect_ratios = np.linspace(0.6, 1.6, 11)
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2 * aspect / (1 + aspect)
        h = 2 / (1 + aspect)
        
        # Initialize with a grid
        # Estimate optimal grid size
        # For aspect ratio A, width is A*height.
        # Let's just use a 5x5 grid scaled to fit
        side = int(np.ceil(np.sqrt(N)))
        
        # Create initial positions
        # We want to fit N circles.
        # Let's place them in a grid that fits in w x h
        # Number of cols and rows
        cols = side
        rows = side
        
        # Scale grid to fit inside w x h with some margin
        # Actually, let's just place them in the center
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Initial radius estimate
        # Area of box = w*h. Area of circles = N * pi * r^2.
        # Packing density ~ 0.9.
        # N * pi * r^2 * 0.9 <= w * h
        # r <= sqrt(w*h / (N * pi * 0.9))
        r_init = np.sqrt(w * h / (N * np.pi * 0.9))
        
        # Grid positions
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                # Position in grid
                px = (j + 0.5) * (w / cols)
                py = (i + 0.5) * (h / rows)
                x[idx] = px
                y[idx] = py
                r[idx] = r_init
                idx += 1
        
        # If N doesn't fit exactly in rows*cols, we might have empty spots or need to adjust.
        # For N=21, 5x5=25. We use 21.
        
        # Local search
        # We will try to increase radii and resolve conflicts.
        
        x_cur = x.copy()
        y_cur = y.copy()
        r_cur = r.copy()
        
        # Optimization loop
        # Use a simple iterative method:
        # 1. Try to increase all radii by dr.
        # 2. If conflicts exist, resolve them by moving centers and reducing radii.
        # 3. Repeat.
        
        dr = 0.001
        steps = 1000
        converged = False
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Try to grow
            r_new = r_cur + dr
            # Check if growth is feasible
            feasible = True
            # Check bounds
            if np.any(x_cur - r_new < -1e-6) or np.any(x_cur + r_new > w + 1e-6) or \
               np.any(y_cur - r_new < -1e-6) or np.any(y_cur + r_new > h + 1e-6):
                # Cannot grow uniformly, maybe reduce dr or stop
                # For simplicity, if we can't grow, we try to move to free space
                # But for now, let's just resolve overlaps first
                pass
            
            # Check overlaps
            # We will use a simple relaxation method
            # For each pair, if overlap, push apart and shrink
            # This is O(N^2) per step. N=21, so 441 pairs. 1000 steps is fine.
            
            # Apply repulsion and shrinkage
            for _ in range(5): # Inner iterations for stability
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x_cur[j] - x_cur[i]
                        dy = y_cur[j] - y_cur[i]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = r_cur[i] + r_cur[j]
                        
                        if dist < min_dist:
                            # Overlap
                            # Push apart
                            if dist < 1e-6:
                                dx = 0.01
                                dy = 0.0
                                dist = 0.01
                            
                            overlap = min_dist - dist
                            # Move each by half the overlap
                            # Also shrink radii slightly to help
                            push = overlap * 0.5
                            x_cur[i] -= (dx / dist) * push
                            y_cur[i] -= (dy / dist) * push
                            x_cur[j] += (dx / dist) * push
                            y_cur[j] += (dy / dist) * push
                            
                            # Shrink radii
                            shrink = overlap * 0.1
                            r_cur[i] -= shrink
                            r_cur[j] -= shrink
                            if r_cur[i] < 1e-4: r_cur[i] = 1e-4
                            if r_cur[j] < 1e-4: r_cur[j] = 1e-4
                            
                # Check bounds and push in
                for i in range(N):
                    if x_cur[i] - r_cur[i] < 0:
                        x_cur[i] = r_cur[i]
                    if x_cur[i] + r_cur[i] > w:
                        x_cur[i] = w - r_cur[i]
                    if y_cur[i] - r_cur[i] < 0:
                        y_cur[i] = r_cur[i]
                    if y_cur[i] + r_cur[i] > h:
                        y_cur[i] = h - r_cur[i]
            
            # After resolving, try to grow again
            # Check if we can grow
            can_grow = True
            for i in range(N):
                if x_cur[i] - r_cur[i] < 1e-6 or x_cur[i] + r_cur[i] > w - 1e-6 or \
                   y_cur[i] - r_cur[i] < 1e-6 or y_cur[i] + r_cur[i] > h - 1e-6:
                    # If touching boundary, we can't grow this circle much without moving
                    # But we can still try
                    pass
                
            # Simple growth: increase all radii by dr if no overlaps after growth
            # To save time, we just add dr and let the next iteration fix overlaps
            r_cur += dr
            
            # Ensure radii are positive
            r_cur = np.maximum(r_cur, 1e-4)
            
            # Check if we are stuck (no growth in last few steps)
            # For now, just run fixed steps
            
        # Final polish: ensure validity
        # We might have invalid configurations if the relaxation failed.
        # Let's verify and fix
        valid = True
        for i in range(N):
            if x_cur[i] - r_cur[i] < -1e-5 or x_cur[i] + r_cur[i] > w + 1e-5 or \
               y_cur[i] - r_cur[i] < -1e-5 or y_cur[i] + r_cur[i] > h + 1e-5:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_cur[j] - x_cur[i]
                    dy = y_cur[j] - y_cur[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_cur[i] + r_cur[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            sum_r = np.sum(r_cur)
            if sum_r > best_sum:
                best_sum = sum_r
                best_result = np.column_stack((x_cur, y_cur, r_cur))
        
    # If no valid result found, return the seed grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
