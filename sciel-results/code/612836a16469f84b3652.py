"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def initial_grid2():
    """Create a slightly perturbed 4x4 grid."""
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x * 2, y * 2])
    return points

def initial_grid3():
    """Create a 4x4 grid with one point moved."""
    points = initial_grid()
    points[0] = [1, 1]  # Move (0,0) to (1,1) - creates a cluster
    points[1] = [-1, 0]
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Try multiple initial configurations
    best_points = None
    best_score = 0
    
    initial_configs = [
        initial_grid(),
        initial_grid2(),
        initial_grid3(),
        # Add a random initial configuration
        [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
    ]
    
    # Remove duplicates from initial configs
    seen = set()
    unique_configs = []
    for cfg in initial_configs:
        key = tuple(tuple(p) for p in sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    
    for init_cfg in unique_configs:
        if time.time() > deadline:
            break
        
        score, has_five = count_4_lines(init_cfg)
        if not has_five and score > best_score:
            best_score = score
            best_points = [p[:] for p in init_cfg]
    
    if best_points is None:
        best_points = initial_grid()
        best_score, _ = count_4_lines(best_points)
    
    # Simulated Annealing with improved parameters
    current_points = [p[:] for p in best_points]
    current_score, _ = count_4_lines(current_points)
    
    temp = 5.0
    cooling = 0.998
    min_temp = 0.01
    max_iter = 100000
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose move type
        r = random.random()
        if r < 0.5:
            # Small jitter
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        elif r < 0.75:
            # Medium jitter
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        elif r < 0.9:
            # Large jump
            idx = random.randint(0, 15)
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        else:
            # Swap two points
            idx1, idx2 = random.sample(range(16), 2)
            current_points[idx1], current_points[idx2] = current_points[idx2], current_points[idx1]
            new_score, has_five = count_4_lines(current_points)
            if has_five:
                # Revert swap
                current_points[idx1], current_points[idx2] = current_points[idx2], current_points[idx1]
                temp *= cooling
                continue
            
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                delta = current_score - new_score
                if random.random() < pow(2.718281828, -delta / temp):
                    current_score = new_score
                else:
                    # Revert swap
                    current_points[idx1], current_points[idx2] = current_points[idx2], current_points[idx1]
            
            temp *= cooling
            continue
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            temp *= cooling
            continue
        
        # Check distinctness
        new_pt = [new_x, new_y]
        is_valid_distinct = True
        for k in range(16):
            if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                is_valid_distinct = False
                break
        
        if not is_valid_distinct:
            temp *= cooling
            continue
        
        # Apply move
        old_pt = current_points[idx][:]
        current_points[idx] = new_pt
        
        new_score, has_five = count_4_lines(current_points)
        
        if has_five:
            # Reject moves that create 5 collinear
            current_points[idx] = old_pt
            temp *= cooling
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
        
        if temp < min_temp:
            break
    
    return best_points
# EVOLVE-BLOCK-END
