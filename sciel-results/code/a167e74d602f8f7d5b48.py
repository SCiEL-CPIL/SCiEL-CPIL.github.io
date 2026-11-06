"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from math import gcd
from itertools import combinations

def count_4_lines_fast(points):
    """Count lines with exactly 4 points using vectorized approach.
    Returns (count, has_five_collinear)."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    # Generate all pair combinations
    pairs = list(combinations(range(n), 2))
    
    # For each pair, compute the line and count points on it
    line_keys = set()
    has_five = False
    
    for i, j in pairs:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            continue
        
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        
        # Canonical direction
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n, dy_n = -dx_n, -dy_n
        
        # Count points on this line: (x - x1) * dy_n == (y - y1) * dx_n
        diffs_x = pts[:, 0] - x1
        diffs_y = pts[:, 1] - y1
        on_line = (diffs_x * dy_n == diffs_y * dx_n)
        cnt = int(np.sum(on_line))
        
        if cnt >= 5:
            return 0, True
        
        if cnt >= 4:
            # Line key: (dx_n, dy_n, c) where c = dy_n * x1 - dx_n * y1
            c = dy_n * x1 - dx_n * y1
            line_keys.add((dx_n, dy_n, c))
    
    return len(line_keys), False

def generate_random_config(scale=200):
    """Generate a random configuration of 16 points."""
    points = []
    used = set()
    for _ in range(16):
        while True:
            x = random.randint(-scale, scale)
            y = random.randint(-scale, scale)
            if (x, y) not in used:
                used.add((x, y))
                points.append([x, y])
                break
    return points

def generate_grid_config():
    """Generate a 4x4 grid configuration."""
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x, y])
    return points

def generate_perturbed_grid(scale=10):
    """Generate a perturbed 4x4 grid."""
    points = generate_grid_config()
    for i in range(16):
        if random.random() < 0.3:
            points[i][0] += random.randint(-scale, scale)
            points[i][1] += random.randint(-scale, scale)
    # Ensure distinctness
    seen = set()
    new_points = []
    for p in points:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            new_points.append(p)
        else:
            # Find a new position
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    new_points.append([x, y])
                    break
    return new_points

def simulated_annealing(initial_points, deadline, start_temp=3.0, cooling=0.9995, min_temp=0.05):
    """Run simulated annealing from initial configuration."""
    current_points = [p[:] for p in initial_points]
    current_score, has_five = count_4_lines_fast(current_points)
    
    if has_five:
        # Try to fix by moving a random point
        for _ in range(100):
            idx = random.randint(0, 15)
            current_points[idx] = [random.randint(-200, 200), random.randint(-200, 200)]
            current_score, has_five = count_4_lines_fast(current_points)
            if not has_five:
                break
        if has_five:
            return initial_points, 0
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    temp = start_temp
    
    while temp > min_temp:
        if time.time() > deadline - 1.0:
            break
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Small local move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        elif move_type < 0.7:
            # Medium jump
            idx = random.randint(0, 15)
            new_x = random.randint(-300, 300)
            new_y = random.randint(-300, 300)
        else:
            # Large jump (restart-like)
            idx = random.randint(0, 15)
            new_x = random.randint(-800, 800)
            new_y = random.randint(-800, 800)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check if point is already used
        new_pt = [new_x, new_y]
        duplicate = False
        for k, p in enumerate(current_points):
            if k != idx and p[0] == new_x and p[1] == new_y:
                duplicate = True
                break
        if duplicate:
            continue
        
        # Apply move
        old_pt = current_points[idx][:]
        current_points[idx] = new_pt
        new_score, has_five = count_4_lines_fast(current_points)
        
        if has_five:
            # Reject and try again
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
            if delta == 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / temp)
            if not accept:
                current_points[idx] = old_pt
        
        temp *= cooling
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    global_best_points = generate_grid_config()
    global_best_score, _ = count_4_lines_fast(global_best_points)
    
    # Multiple restarts with different initial configurations
    num_restarts = 8
    per_restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * per_restart_time
        if time.time() > restart_deadline - 0.5:
            break
        
        # Choose initial configuration
        if restart == 0:
            init_pts = generate_grid_config()
        elif restart == 1:
            init_pts = generate_perturbed_grid(scale=5)
        elif restart == 2:
            init_pts = generate_perturbed_grid(scale=20)
        elif restart == 3:
            init_pts = generate_random_config(scale=100)
        elif restart == 4:
            init_pts = generate_random_config(scale=300)
        elif restart == 5:
            init_pts = generate_random_config(scale=500)
        elif restart == 6:
            init_pts = generate_perturbed_grid(scale=50)
        else:
            init_pts = generate_random_config(scale=200)
        
        # Run simulated annealing
        best_pts, best_score = simulated_annealing(init_pts, restart_deadline)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_points = best_pts
    
    return global_best_points
# EVOLVE-BLOCK-END
