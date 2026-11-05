import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Initial configuration: 4x4 grid
    # This has score 10.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        # Normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # The line is defined by direction (dx, dy) and a point (x1, y1).
        # To make it unique, we can use the cross product or intercept.
        # Equation: dy*x - dx*y = C
        # C = dy*x1 - dx*y1
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
        # Check for distinctness first
        if len(set(tuple(p) for p in points)) != len(points):
            return -10000
            
        # Check for 5+ collinear and count 4-collinear lines
        n = len(points)
        line_counts = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_counts:
                    line_counts[key] = 2
                else:
                    line_counts[key] += 1
        
        # A line with k points contributes C(k,2) to the sum of counts above.
        # We need to identify lines with exactly 4 points.
        # If a line has 4 points, it appears C(4,2)=6 times.
        # If a line has 5 points, it appears C(5,2)=10 times.
        # So we can just look at the counts.
        # However, multiple lines can share pairs? No, a pair defines a unique line.
        # So the keys in line_counts are unique lines.
        # The value is the number of pairs on that line.
        # If value is 6, it's a 4-point line.
        # If value is 10, it's a 5-point line.
        # If value is 15, it's a 6-point line.
        
        score = 0
        invalid = False
        for count in line_counts.values():
            if count == 6:
                score += 1
            elif count >= 10:
                invalid = True
                break
            # count == 3 means 3 points (C(3,2)=3)
            # count == 1 means 2 points (C(2,2)=1) - wait, C(2,2)=1.
            # So count is the number of pairs.
            # k points -> C(k,2) pairs.
            # k=2 -> 1
            # k=3 -> 3
            # k=4 -> 6
            # k=5 -> 10
            # k=6 -> 15
            
        if invalid:
            return -1000 + score # Penalize invalidity
        return score

    best_points = [p[:] for p in initial_points]
    best_score = evaluate(best_points)
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Move by a small random amount
        # Range -5 to 5 is good for local search
        new_x = old_x + random.randint(-5, 5)
        new_y = old_y + random.randint(-5, 5)
        
        # Keep within bounds |x|,|y| <= 1000
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check distinctness
        new_point = [new_x, new_y]
        is_distinct = True
        for k in range(16):
            if k != idx and current_points[k] == new_point:
                is_distinct = False
                break
        
        if not is_distinct:
            continue

        current_points[idx] = new_point
        new_score = evaluate(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Simulated Annealing acceptance
            delta = current_score - new_score
            if delta < 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / T)
            
            if accept:
                # Keep the move
                pass
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
        T *= 0.999
        if T < T_min:
            T = T_min
            # Restart or just continue?
            # If stuck, maybe reheat?
            # For now, just continue.
            
    return best_points
