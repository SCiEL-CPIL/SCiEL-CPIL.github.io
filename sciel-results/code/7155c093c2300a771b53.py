import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial seed: 6x5 grid
    # Points in [-30, 30]
    # Center it to stay within bounds? The grid 0..5, 0..4 is within [-30, 30].
    def get_initial_grid(w, h):
        return [(x, y) for x in range(w) for y in range(h)]
    
    best_points = get_initial_grid(6, 5)
    
    def score(points):
        """Compute the maximum multiplicity of any single positive doubled area."""
        n = len(points)
        cnt = Counter()
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                for k in range(j + 1, n):
                    xk, yk = points[k]
                    d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    if d > 0:
                        cnt[d] += 1
        return max(cnt.values()) if cnt else 0
    
    # Evaluate seed
    best_score = score(best_points)
    
    # Search
    # We will use a simple random perturbation strategy
    # Move one point to a random location in [-30, 30]
    # Ensure distinctness
    
    for _ in range(20000): # Max iterations, will break on time
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 29)
        # Pick a new random position
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        new_point = (new_x, new_y)
        
        # Check if new point is already in the set (excluding the one being moved)
        if new_point in best_points[:idx] + best_points[idx+1:]:
            continue
            
        # Create new candidate
        candidate = best_points.copy()
        candidate[idx] = new_point
        
        # Evaluate
        s = score(candidate)
        
        if s > best_score:
            best_score = s
            best_points = candidate
        
    return best_points
