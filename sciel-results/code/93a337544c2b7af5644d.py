import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        n = len(pts)
        # Check for validity: no 5 collinear
        # And count lines with exactly 4
        line_counts = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize direction vector
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Ensure canonical form
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Line key: (dx, dy, x1*dx - y1*dy) ? 
                # Better: use (dx, dy, c) where line is dx*x - dy*y = c
                # c = dx*x1 - dy*y1
                c = dx * x1 - dy * y1
                key = (dx, dy, c)
                if key not in line_counts:
                    line_counts[key] = 2
                else:
                    line_counts[key] += 1
        
        score = 0
        for key, count in line_counts.items():
            if count > 4:
                return -1000 # Invalid
            if count == 4:
                score += 1
        return score

    best_points = points
    best_score = get_score(points)
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random point and move it
        idx = random.randint(0, 15)
        old_p = current_points[idx]
        # Random move: small step or large jump
        if random.random() < 0.8:
            new_x = old_p[0] + random.randint(-2, 2)
            new_y = old_p[1] + random.randint(-2, 2)
        else:
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        is_distinct = True
        for i, p in enumerate(current_points):
            if i == idx:
                continue
            if p[0] == new_x and p[1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue

        current_points[idx] = [new_x, new_y]
        new_score = get_score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    current_points[idx] = old_p
            # If delta >= 0, already accepted (though handled above)
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_points
