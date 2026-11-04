import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_score = 0
    best_config = None
    
    # Scan aspect ratios
    # w + h = 2
    # w in [0.5, 1.5]
    for w in np.linspace(0.6, 1.4, 17):
        h = 2.0 - w
        
        # Initial guess: 5x4 grid
        # 5 columns, 4 rows
        cols = 5
        rows = 4
        # Add 1 extra circle? 20 in grid, 1 extra.
        # Let's just use a 5x4 grid and see if we can squeeze the 21st.
        # Or use a 6x4 grid? 24 slots.
        # Let's try to initialize 21 circles in a grid pattern.
        
        # Simple initialization:
        # Place 20 in 5x4 grid, 1 in the middle of the last row?
        # Or just 21 in a 5x4.2 grid?
        
        # Let's use a more robust initialization:
        # Distribute 21 points in w x h
        xs = []
        ys = []
        # 5 columns
        for i in range(5):
            x = (i + 0.5) * w / 5
            if i < 4:
                # 4 rows for first 4 columns? No.
                pass
        
        # Better initialization:
        # 21 circles.
        # Try 4 rows of 5, and 1 row of 1?
        # Or 5 rows of 4, and 1 row of 1?
        
        # Let's just use a random initialization with a few restarts for each aspect ratio.
        for restart in range(3):
            if time.time() > deadline:
                break
                
            # Initialize positions
            if restart == 0:
                # Grid 5x4 + 1
                xs = []
                ys = []
                for i in range(5):
                    for j in range(4):
                        xs.append((i + 0.5) * w / 5)
                        ys.append((j + 0.5) * h / 4)
                # 21st circle
                xs.append(w / 2)
                ys.append(h / 2)
            else:
                # Random
                xs = np.random.uniform(0, w, N)
                ys = np.random.uniform(0, h, N)
            
            # Initial radius
            r = 0.05
            # Ensure inside
            xs = np.clip(xs, r, w - r)
            ys = np.clip(ys, r, h - r)
            
            # Local search
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Check constraints
                # 1. Inside rectangle
                # 2. No overlap
                
                # Calculate overlap penalty
                penalty = 0
                grad_x = np.zeros(N)
                grad_y = np.zeros(N)
                
                # Wall constraints
                for i in range(N):
                    if xs[i] < r:
                        penalty += (r - xs[i])**2
                        grad_x[i] -= 2 * (r - xs[i])
                    elif xs[i] > w - r:
                        penalty += (xs[i] - (w - r))**2
                        grad_x[i] += 2 * (xs[i] - (w - r))
                    
                    if ys[i] < r:
                        penalty += (r - ys[i])**2
                        grad_y[i] -= 2 * (r - ys[i])
                    elif ys[i] > h - r:
                        penalty += (ys[i] - (h - r))**2
                        grad_y[i] += 2 * (ys[i] - (h - r))
                
                # Overlap constraints
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = 2 * r
                        if dist_sq < min_dist*min_dist:
                            dist = np.sqrt(dist_sq)
                            if dist > 1e-6:
                                # Gradient of (dist - min_dist)^2
                                # d/dx (dist^2 - min_dist^2)^2 ?
                                # Let's use penalty: (min_dist^2 - dist_sq)^2 if dist_sq < min_dist^2
                                diff = min_dist*min_dist - dist_sq
                                penalty += diff*diff
                                # Gradient
                                # d/dx_i (min_dist^2 - (xi-xj)^2 - (yi-yj)^2)^2
                                # = 2 * (min_dist^2 - dist_sq) * (-2 * (xi-xj))
                                grad_x[i] += 4 * diff * dx
                                grad_y[i] += 4 * diff * dy
                                grad_x[j] -= 4 * diff * dx
                                grad_y[j] -= 4 * diff * dy
                            else:
                                # Random push
                                grad_x[i] += np.random.randn()
                                grad_y[i] += np.random.randn()
                                grad_x[j] -= np.random.randn()
                                grad_y[j] -= np.random.randn()
                
                if penalty < 1e-6:
                    # Try to increase r
                    r += 0.001
                    # Re-clamp
                    xs = np.clip(xs, r, w - r)
                    ys = np.clip(ys, r, h - r)
                else:
                    # Move to reduce penalty
                    # Learning rate
                    lr = 0.01
                    xs += lr * grad_x
                    ys += lr * grad_y
                    # Clamp to rectangle (with current r)
                    xs = np.clip(xs, r, w - r)
                    ys = np.clip(ys, r, h - r)
                
                # Check score
                score = N * r
                if score > best_score:
                    best_score = score
                    best_config = (xs.copy(), ys.copy(), r, w, h)
            
            # If we found a valid config with high r, break restarts
            if best_score > N * 0.11:
                break
    
    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    xs, ys, r, w, h = best_config
    result = np.column_stack((xs, ys, np.full(N, r)))
    return result

