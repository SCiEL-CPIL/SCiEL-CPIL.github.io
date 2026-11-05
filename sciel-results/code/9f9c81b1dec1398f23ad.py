"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        # Equation: dy*x - dx*y = C
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def evaluate(points):
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
        
        score = 0
        invalid = False
        for count in line_counts.values():
            # count is number of pairs.
            # k points -> C(k,2) pairs.
            # k=4 -> 6
            # k=5 -> 10
            if count == 6:
                score += 1
            elif count >= 10:
                invalid = True
                break
            
        if invalid:
            return -1000 + score
        return score

    best_points = [p[:] for p in initial_points]
    best_score = evaluate(best_points)
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    step_size = 2
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Move by a small random amount
        new_x = old_x + random.randint(-step_size, step_size)
        new_y = old_y + random.randint(-step_size, step_size)
        
        # Keep within bounds |x|,|y| <= 1000
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check for duplicate
        is_dup = False
        for k, p in enumerate(current_points):
            if k != idx and p[0] == new_x and p[1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue

        current_points[idx] = [new_x, new_y]
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
            # Occasionally increase step size if stuck?
            # Simple heuristic: if no improvement in last 1000 steps, increase step size
            # We don't track steps here, so just let it run.
            
    return best_points
# EVOLVE-BLOCK-END
