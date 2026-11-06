import time
import random
from math import gcd
from itertools import combinations

def count_4_lines(points):
    """
    Counts lines with exactly 4 points.
    Returns (count, has_five_collinear).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    line_pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Canonical sign: dx > 0 or dx == 0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Line constant c: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    count_4 = 0
    has_5 = False
    
    for cnt in line_pair_counts.values():
        # cnt is C(k, 2)
        if cnt == 6:
            count_4 += 1
        elif cnt >= 10:
            has_5 = True
            return 0, True
            
    return count_4, has_5

def initial_grid():
    """Create a 4x4 grid as starting point."""
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x, y])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = initial_grid()
    best_score, _ = count_4_lines(best_points)
    
    # Run multiple SA restarts
    num_restarts = 50
    time_per_restart = 140.0 / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize current state
        # Sometimes start with a slightly perturbed grid to escape symmetry
        current_points = [p[:] for p in best_points]
        if r > 0 and random.random() < 0.5:
             # Perturb a few points
             for _ in range(random.randint(1, 3)):
                 idx = random.randint(0, 15)
                 current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        current_score, has_five = count_4_lines(current_points)
        if has_five:
             # Revert if invalid
             current_points = [p[:] for p in best_points]
             current_score, _ = count_4_lines(current_points)
        
        # SA Parameters
        temp = 5.0
        cooling = 0.99
        min_temp = 0.01
        restart_deadline = time.time() + time_per_restart
        
        while temp > min_temp:
            if time.time() > restart_deadline:
                break
            
            # Choose move type
            if random.random() < 0.8:
                # Local move
                idx = random.randint(0, 15)
                old_x, old_y = current_points[idx]
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
            else:
                # Global jump
                idx = random.randint(0, 15)
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            # Check distinctness
            new_pt = [new_x, new_y]
            is_valid_distinct = True
            for k in range(16):
                if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                    is_valid_distinct = False
                    break
            
            if not is_valid_distinct:
                continue
            
            # Apply move
            old_pt = current_points[idx][:]
            current_points[idx] = new_pt
            
            new_score, has_five = count_4_lines(current_points)
            
            if has_five:
                # Reject moves that create 5 collinear
                current_points[idx] = old_pt
                continue
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept with probability exp(-delta/temp)
                delta = current_score - new_score
                if random.random() < pow(2.718281828, -delta / temp):
                    current_score = new_score
                else:
                    # Revert
                    current_points[idx] = old_pt
            
            temp *= cooling
            
    # Final Polish: Intensive local search on the best found
    if time.time() < deadline - 5:
        current_points = [p[:] for p in best_points]
        current_score, _ = count_4_lines(current_points)
        
        # Very fine-grained search
        for _ in range(10000):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-1, 1)
            new_y = old_y + random.randint(-1, 1)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            new_pt = [new_x, new_y]
            is_valid_distinct = True
            for k in range(16):
                if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                    is_valid_distinct = False
                    break
            
            if not is_valid_distinct:
                continue
            
            old_pt = current_points[idx][:]
            current_points[idx] = new_pt
            new_score, has_five = count_4_lines(current_points)
            
            if has_five:
                current_points[idx] = old_pt
                continue
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = old_pt

    return best_points
