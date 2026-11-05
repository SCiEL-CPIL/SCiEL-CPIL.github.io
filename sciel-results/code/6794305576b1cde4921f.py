import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def _calculate_max_multiplicity(points):
        n = len(points)
        if n < 3:
            return 0, 0
        areas = {}
        # Vectorized calculation for speed
        # Create arrays of points
        pts = np.array(points)
        # We need to check all triples (i, j, k) with i < j < k
        # To keep it fast, we iterate i, then vectorize j, k? 
        # Actually, simple loops in Python might be slower than expected for 30 points (4060 triples).
        # 4060 determinants is trivial. The bottleneck is the loop overhead in Python if not optimized.
        # Let's use a slightly optimized loop or numpy broadcasting if possible.
        # For 30 points, a triple loop is 30*29*28/6 = 4060 iterations. This is very fast.
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        areas[det] = areas.get(det, 0) + 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        max_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, max_area

    def _evaluate(points):
        return _calculate_max_multiplicity(points)[0]

    # Initialize with a good grid configuration
    # 5x6 grid centered
    w, h = 5, 6
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append((x - w//2, y - h//2))
    
    current_points = [tuple(p) for p in pts]
    current_score = _evaluate(current_points)
    
    best_points = list(current_points)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 100.0
    cooling_rate = 0.999
    min_T = 0.1
    
    random.seed(42)
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        # Generate a neighbor
        # Strategy: Pick a random point and move it to a random valid location
        # Or pick a random point and shift it by a random amount
        idx = random.randint(0, 29)
        old_pos = current_points[idx]
        
        # Random jump: choose a new position within bounds
        # To ensure diversity, sometimes do a small step, sometimes a big jump
        if random.random() < 0.5:
            # Small step
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        else:
            # Big jump
            new_pos = (random.randint(-30, 30), random.randint(-30, 30))
        
        # Check bounds and uniqueness
        if not (-30 <= new_pos[0] <= 30 and -30 <= new_pos[1] <= 30):
            continue
        if new_pos in current_points:
            continue
            
        # Apply move
        new_points = list(current_points)
        new_points[idx] = new_pos
        
        new_score = _evaluate(new_points)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
        else:
            # Accept worse state with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_points = new_points
                current_score = new_score
                
        T *= cooling_rate
        
        # Occasionally reheat to escape deep local optima
        if steps % 1000 == 0 and T < 1.0:
            T = 10.0

    return best_points
