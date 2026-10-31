"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_result = None
    
    # Multiple strategies for initialization
    def grid_init(spacing_factor=1.0):
        side = int(np.ceil(np.sqrt(N)))
        # Adjust spacing to fit N circles roughly
        # For N=26, 5x5 grid is 25, so we need one extra or 6x5?
        # Let's just use a 5x5 grid and one in the middle? No, 5x5=25.
        # Let's use a 5x6 grid? 30 circles. Use first 26.
        # Or a hexagonal packing.
        pass

    def hex_init():
        # Hexagonal packing is denser
        circles = []
        r = 0.05
        dy = r * np.sqrt(3)
        dx = 2 * r
        row = 0
        while len(circles) < N:
            for col in range(10):
                x = (col + 0.5 * (row % 2)) * dx + r
                y = row * dy + r
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, r])
            row += 1
        return np.array(circles[:N])

    def random_init():
        # Random positions, small radius
        r = 0.02
        x = np.random.uniform(r, 1-r, N)
        y = np.random.uniform(r, 1-r, N)
        return np.column_stack((x, y, np.full(N, r)))

    def grow_and_push(circles, max_iters=5000):
        # circles: Nx3 array
        # Returns final circles and score
        r_step = 0.001
        pos_step = 0.005
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # 1. Grow radii
            circles[:, 2] += r_step
            
            # 2. Resolve constraints
            # Iterate over circles and push them
            # To avoid bias, we can randomize order or use simultaneous update
            # Simple sequential push:
            for i in range(N):
                x, y, r = circles[i]
                
                # Boundary constraints
                if x - r < 0: x = r
                if x + r > 1: x = 1 - r
                if y - r < 0: y = r
                if y + r > 1: y = 1 - r
                
                # Overlap constraints
                for j in range(N):
                    if i == j: continue
                    xj, yj, rj = circles[j]
                    dx = x - xj
                    dy = y - yj
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r + rj
                    
                    if dist < min_dist:
                        # Push i away from j
                        if dist > 1e-9:
                            push_x = (dx / dist) * (min_dist - dist)
                            push_y = (dy / dist) * (min_dist - dist)
                            x += push_x
                            y += push_y
                        else:
                            # Random push
                            x += np.random.uniform(-pos_step, pos_step)
                            y += np.random.uniform(-pos_step, pos_step)
                
                circles[i, 0] = x
                circles[i, 1] = y

            # Check if we are stuck (optional, for early exit or perturbation)
            # For now, just keep going
            
        return circles

    def evaluate(circles):
        # Check validity
        for i in range(N):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
                return -1.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if dist < ri + rj - 1e-6:
                    return -1.0
        return np.sum(circles[:, 2])

    # Run multiple restarts with different inits
    inits = [
        lambda: grid_init(),
        lambda: hex_init(),
        lambda: random_init()
    ]
    
    # We need a proper grid_init
    def grid_init_impl():
        # 5x5 grid is 25. We need 26.
        # Let's do 5 rows, 5 cols, and put the 26th in the center?
        # Or just 6x5 grid and take 26.
        side_x = 5
        side_y = 5
        # This leaves 1 spot.
        # Let's just use a 5x5 grid of circles, and one smaller one?
        # No, radii must be maximized.
        # Let's use a 5x5 grid, but slightly perturb to allow the 26th?
        # Actually, a 5x5 grid of equal circles has radius ~ 1/10.
        # Sum = 25 * 0.1 = 2.5.
        # If we pack 26, average radius might be slightly less, but sum could be higher?
        # No, adding a circle usually forces others to shrink.
        # Wait, the task is to pack 26 circles.
        # A 5x5 grid has 25.
        # We can try a 5x5 grid and squeeze in one more?
        # Or use a 6x5 grid (30 circles) and remove 4?
        # Let's start with a 6x5 grid, remove 4 corners, and grow.
        circles = []
        r = 0.05
        for i in range(6):
            for j in range(5):
                x = (i + 0.5) / 6.0
                y = (j + 0.5) / 5.0
                circles.append([x, y, r])
        # Remove 4 circles to get 26?
        # Let's just use the first 26.
        return np.array(circles[:N])

    inits[0] = grid_init_impl

    for init_func in inits:
        if time.time() > deadline - 10.0:
            break
            
        # Try a few times with each init
        for k in range(5):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(k * 100 + 42) # Deterministic-ish restarts
            try:
                c = init_func()
                # Ensure valid start
                if evaluate(c) < 0:
                    # Shrink radii
                    c[:, 2] *= 0.5
                    if evaluate(c) < 0:
                        continue
                        
                c = grow_and_push(c)
                score = evaluate(c)
                
                if score > best_score:
                    best_score = score
                    best_result = c.copy()
                    
            except Exception:
                pass

    # If we have time, try to refine the best result
    if best_result is not None and time.time() < deadline - 20.0:
        # Simple local search: small random perturbations + grow and push
        current_best = best_result.copy()
        current_score = best_score
        
        for _ in range(100): # Max refinement loops
            if time.time() > deadline - 10.0:
                break
            
            # Perturb
            c = current_best.copy()
            c[:, :2] += np.random.normal(0, 0.01, (N, 2))
            
            # Clip to box
            c[:, 0] = np.clip(c[:, 0], c[:, 2], 1 - c[:, 2])
            c[:, 1] = np.clip(c[:, 1], c[:, 2], 1 - c[:, 2])
            
            # Grow and push from this state
            c = grow_and_push(c, max_iters=1000)
            
            score = evaluate(c)
            if score > current_score:
                current_score = score
                current_best = c.copy()
                best_score = current_score
                best_result = current_best.copy()
            else:
                # If no improvement, maybe break or continue
                pass

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result

# EVOLVE-BLOCK-END
