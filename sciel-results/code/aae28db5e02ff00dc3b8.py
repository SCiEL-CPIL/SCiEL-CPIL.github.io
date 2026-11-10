import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # Initial candidates: Rectangular grids
    # We want 30 points.
    # Factors of 30: 1, 2, 3, 5, 6, 10, 15, 30
    # Grids (w, h)
    grid_shapes = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2)]
    
    best_score = 0
    best_config = None
    
    # Helper to score a configuration
    def score_config(points):
        # points: list of (x, y)
        n = len(points)
        if n < 3:
            return 0
        # Use numpy for speed if possible, but loop is fine for N=30
        # To be safe and fast, we can use a dictionary for counts
        counts = {}
        # Precompute vectors? No, just compute determinants
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] = counts.get(det, 0) + 1
        if not counts:
            return 0
        return max(counts.values())
    
    # Initialize with best grid
    for w, h in grid_shapes:
        # Center the grid in [-30, 30]
        # Range 0..w-1, 0..h-1
        # Shift to center: 
        # x_center = -w//2, y_center = -h//2
        start_x = -w // 2
        start_y = -h // 2
        points = [(start_x + i, start_y + j) for i in range(w) for j in range(h)]
        # Check bounds [-30, 30]
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in points):
            s = score_config(points)
            if s > best_score:
                best_score = s
                best_config = points[:]
        
    # If no valid grid found (unlikely), use random
    if best_config is None:
        best_config = [(random.randint(-30, 30), random.randint(-30, 30)) for _ in range(30)]
        # Ensure distinct
        while len(set(best_config)) < 30:
             best_config[random.randint(0, 29)] = (random.randint(-30, 30), random.randint(-30, 30))
        best_score = score_config(best_config)
    
    current_config = best_config[:]
    current_score = best_score
    
    # Local Search
    # We try to move a single point to a random nearby or random position
    # To keep it fast, we'll use a simple hill climbing with random restarts if stuck?
    # Or just continuous hill climbing.
    
    # Optimization: 
    # Instead of random anywhere, maybe random nearby is better for grids?
    # But random anywhere allows escaping local minima.
    # Let's mix: 50% random nearby (radius 5), 50% random anywhere.
    
    while time.time() - start_time < deadline:
        # Pick a point to move
        idx = random.randint(0, 29)
        old_point = current_config[idx]
        
        # Generate new point
        if random.random() < 0.5:
            # Nearby
            new_x = old_point[0] + random.randint(-5, 5)
            new_y = old_point[1] + random.randint(-5, 5)
        else:
            # Anywhere
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        
        new_point = (new_x, new_y)
        
        # Check bounds and distinctness
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            continue
        if new_point == old_point:
            continue
            
        # Check if new_point is already in config (excluding idx)
        # This is O(N) which is fine
        is_valid = True
        for i in range(30):
            if i != idx and current_config[i] == new_point:
                is_valid = False
                break
        if not is_valid:
            continue
            
        # Apply move
        current_config[idx] = new_point
        
        # Score
        new_score = score_config(current_config)
        
        if new_score > best_score:
            best_score = new_score
            best_config = current_config[:]
            # Reset current to best? Or keep going? 
            # Keeping going allows further improvement from this new best.
            current_score = new_score
        elif new_score == current_score:
            # Accept sideways move to explore
            current_score = new_score
            # With some probability, accept even if slightly worse? 
            # For "maximize max", worse is bad. 
            # Let's stick to strict improvement or sideways for now.
            # If we get stuck, we might need simulated annealing.
            # But simple hill climbing often works well for these structured problems.
        else:
            # Revert
            current_config[idx] = old_point
            # current_score remains unchanged (it was the score before the move)
            
        # Optional: If we haven't improved in a while, do a jump?
        # For now, keep it simple.
        
    return best_config
