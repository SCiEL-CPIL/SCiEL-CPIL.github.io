import random
import itertools
import math
import numpy as np
from collections import defaultdict

def construct():
    """
    Find 16 distinct integer points with |x|, |y| <= 1000, no five collinear,
    maximizing the number of lines containing exactly four of the points.
    
    Uses simulated annealing with optimized move generation and line counting.
    """
    random.seed(42)
    np.random.seed(42)
    
    def count_lines(points):
        """
        Efficiently count lines with exactly 4 points and check for 5+ collinear.
        Returns (count_4, has_5)
        """
        n = len(points)
        # Map from line key to set of point indices
        line_map = defaultdict(set)
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                
                # Normalize direction
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Line equation: a*x + b*y + c = 0
                a = dy_n
                b = -dx_n
                c = -(a * x1 + b * y1)
                
                # Normalize (a, b, c)
                g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                
                if a < 0:
                    a, b, c = -a, -b, -c
                elif a == 0 and b < 0:
                    a, b, c = -a, -b, -c
                    
                key = (a, b, c)
                line_map[key].add(i)
                line_map[key].add(j)
                
        count_4 = 0
        has_5 = False
        for pts in line_map.values():
            sz = len(pts)
            if sz == 4:
                count_4 += 1
            elif sz >= 5:
                has_5 = True
                # We can break early if we just need validity, but we need count for score.
                # However, if has_5 is True, the move is invalid anyway.
        
        return count_4, has_5

    def is_valid(points):
        """Check distinctness, bounds, and no 5 collinear."""
        if len(set(map(tuple, points))) != 16:
            return False
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return False
        _, has_5 = count_lines(points)
        return not has_5

    def generate_initial():
        """
        Generate a good initial configuration.
        Start with a grid and perturb it to break symmetries that might trap in local optima,
        or just use a random valid config.
        A 4x4 grid is valid and has score 10.
        """
        # Start with 4x4 grid centered at 0
        pts = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                pts.append([x, y])
        
        # Perturb slightly to ensure we don't start in a "perfect" grid which might be a local max
        # but actually grid is a good baseline. Let's add some randomness to break perfect alignment if it helps escape.
        # However, the task is to MAXIMIZE 4-lines. Grid has 10.
        # Maybe a different structure is better.
        # Let's just use the grid as a strong seed.
        return pts

    current_points = generate_initial()
    current_score, _ = count_lines(current_points)
    if not is_valid(current_points):
        # Fallback to random if grid somehow invalid (it shouldn't be)
        while True:
            current_points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
            if is_valid(current_points):
                break
        current_score, _ = count_lines(current_points)

    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    max_iter = 200000 # Increase iterations
    
    for it in range(max_iter):
        if T < T_min:
            break
            
        # Move: Pick a random point and move it to a nearby location
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Local move: small step
        step = random.choice([1, 2, 3, 5, 10])
        new_x = old_x + random.randint(-step, step)
        new_y = old_y + random.randint(-step, step)
        
        # Ensure bounds
        if new_x < -1000: new_x = -1000
        if new_x > 1000: new_x = 1000
        if new_y < -1000: new_y = -1000
        if new_y > 1000: new_y = 1000
        
        # Check collision
        if any(new_x == p[0] and new_y == p[1] for i, p in enumerate(current_points) if i != idx):
            continue
            
        # Apply move
        current_points[idx] = [new_x, new_y]
        
        # Evaluate
        new_score, has_5 = count_lines(current_points)
        
        if has_5:
            # Invalid move, revert
            current_points[idx] = [old_x, old_y]
            T *= cooling
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert if not accepted
            current_points[idx] = [old_x, old_y]
            
        T *= cooling
        
    return best_points
