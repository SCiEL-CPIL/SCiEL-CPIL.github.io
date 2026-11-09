import random
import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    def get_score(pts):
        line_counts = {}
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                a = y2 - y1
                b = x1 - x2
                c = -(a * x1 + b * y1)
                
                # Normalize
                g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g != 0:
                    a //= g
                    b //= g
                    c //= g
                
                # Sign
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                elif a == 0 and b == 0:
                    # Should not happen for distinct points
                    c = 1 if c > 0 else -1
                    
                key = (a, b, c)
                line_counts[key] = line_counts.get(key, 0) + 1
                
        score = 0
        for count in line_counts.values():
            if count == 4:
                score += 1
            elif count >= 5:
                return -1000, False # Invalid
                
        return score, True

    # Initial points: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = points[:]
    best_score, _ = get_score(points)
    current_score = best_score
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.1
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        i = random.randint(0, 15)
        old_p = points[i]
        
        # Move strategy
        r = random.random()
        if r < 0.7:
            # Local move
            step = random.randint(1, 10)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            new_p = [old_p[0] + dx, old_p[1] + dy]
        else:
            # Global move
            new_p = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            
        # Bounds check
        new_p[0] = max(-1000, min(1000, new_p[0]))
        new_p[1] = max(-1000, min(1000, new_p[1]))
        
        # Distinctness check
        if new_p in points:
            continue
            
        # Apply move
        points[i] = new_p
        new_score, is_valid = get_score(points)
        
        if not is_valid:
            points[i] = old_p
            continue
            
        delta = new_score - current_score
        
        # Acceptance criterion
        if new_score > current_score or (T > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            points[i] = old_p
            no_improve_count += 1
            
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Random restart or kick
            # Let's kick: move 3 random points
            for _ in range(3):
                k = random.randint(0, 15)
                points[k] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            new_score, is_valid = get_score(points)
            if is_valid:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points[:]
            else:
                # Revert to best if kick failed badly?
                # Or just revert the kick
                # Simple: revert to best_points and reset T
                points = best_points[:]
                current_score = best_score
                T = 5.0
            no_improve_count = 0

    return best_points
