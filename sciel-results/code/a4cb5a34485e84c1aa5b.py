"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from math import gcd
from itertools import combinations

def count_4_lines(points):
    """Count lines with exactly 4 points.
    Returns (count, has_five_collinear).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    pts = np.array(points, dtype=np.int64)
    line_keys = set()
    has_five = False
    
    # Iterate over pairs to define lines
    # To avoid duplicates, we only consider pairs (i, j) with i < j
    # and canonicalize the line equation.
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Canonical direction: dx_n > 0 or (dx_n == 0 and dy_n > 0)
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            # Line equation: dy_n * x - dx_n * y = c
            # c = dy_n * x1 - dx_n * y1
            c = dy_n * x1 - dx_n * y1
            
            # Check if we've seen this line before
            key = (dx_n, dy_n, c)
            if key in line_keys:
                # We already counted this line. 
                # We need to know if it has 5 points.
                # The set `line_keys` only stores lines with >= 4 points? 
                # No, this logic is flawed if we just add to set.
                # We need to count points per line.
                pass
            
            # Efficient counting:
            # Check all other points k to see if they lie on this line.
            # (x_k - x1) * dy_n == (y_k - y1) * dx_n
            
            # Vectorized check for all points
            # But doing this for every pair is O(N^3). N=16, so 16^3 = 4096 ops per eval.
            # 10000 evals * 4096 = 40M ops. This is fast enough in Python if vectorized?
            # Let's use a simpler loop for clarity and speed in Python for small N.
            
            count = 0
            for k in range(n):
                xk, yk = pts[k]
                if (xk - x1) * dy_n == (yk - y1) * dx_n:
                    count += 1
            
            if count >= 5:
                return 0, True
            
            if count == 4:
                # Only add if not already added
                # We need to ensure we don't double count.
                # The key (dx_n, dy_n, c) uniquely identifies the line.
                line_keys.add(key)
                
    return len(line_keys), False

def count_4_lines_fast(points):
    """Optimized count using numpy for the inner loop."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    line_keys = set()
    has_five = False
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            # Check if already processed
            # We can't just skip if key is in set, because we need to check for 5-collinear
            # But if we already counted it as 4, and now we find it again, it's the same line.
            # If it has 5 points, we would have returned 0, True already if we checked all points.
            # Wait, the previous loop checked ALL points for this pair.
            # If we encounter the same line again from a different pair, we already know its count.
            # So we can skip if key is in a 'processed' set.
            
            if key in line_keys:
                continue
            
            # Count points on this line
            # Vectorized
            diffs_x = pts[:, 0] - x1
            diffs_y = pts[:, 1] - y1
            on_line = (diffs_x * dy_n == diffs_y * dx_n)
            cnt = int(np.sum(on_line))
            
            if cnt >= 5:
                return 0, True
            
            if cnt == 4:
                line_keys.add(key)
            else:
                # If cnt < 4, we don't add it. 
                # But we should mark it as processed to avoid recounting?
                # Yes, if cnt < 4, it will never become 4.
                # But we don't want to add it to line_keys.
                # We can use a separate set for processed lines.
                pass
    
    # Wait, the logic above has a flaw. If cnt < 4, we don't add to line_keys.
    # But if we encounter the same line again, we will recount it.
    # To fix this, we should maintain a set of all processed line keys.
    
    return len(line_keys), False

def count_4_lines_opt(points):
    """Further optimized count."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    line_keys = set()
    processed_keys = set()
    has_five = False
    
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            if key in processed_keys:
                continue
            
            processed_keys.add(key)
            
            # Count points
            diffs_x = pts[:, 0] - x1
            diffs_y = pts[:, 1] - y1
            on_line = (diffs_x * dy_n == diffs_y * dx_n)
            cnt = int(np.sum(on_line))
            
            if cnt >= 5:
                return 0, True
            
            if cnt == 4:
                line_keys.add(key)
                
    return len(line_keys), False

def generate_grid_config():
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x, y])
    return points

def generate_perturbed_grid(scale=10):
    points = generate_grid_config()
    for i in range(16):
        if random.random() < 0.5:
            points[i][0] += random.randint(-scale, scale)
            points[i][1] += random.randint(-scale, scale)
    seen = set()
    new_points = []
    for p in points:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            new_points.append(p)
        else:
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in seen:
                    seen.add((x, y))
                    new_points.append([x, y])
                    break
    return new_points

def generate_random_config(scale=200):
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

def simulated_annealing(initial_points, deadline, start_temp=5.0, cooling=0.999, min_temp=0.1):
    current_points = [p[:] for p in initial_points]
    current_score, has_five = count_4_lines_opt(current_points)
    
    if has_five:
        # Try to fix
        for _ in range(50):
            idx = random.randint(0, 15)
            current_points[idx] = [random.randint(-500, 500), random.randint(-500, 500)]
            current_score, has_five = count_4_lines_opt(current_points)
            if not has_five:
                break
        if has_five:
            return initial_points, 0

    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    temp = start_temp
    
    while temp > min_temp:
        if time.time() > deadline - 0.5:
            break
        
        # Move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random new position
        new_x = old_x + random.randint(-20, 20)
        new_y = old_y + random.randint(-20, 20)
        
        # Clamp
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Check duplicate
        dup = False
        for k in range(16):
            if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                dup = True
                break
        
        if dup:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score, has_five = count_4_lines_opt(current_points)
        
        if has_five:
            current_points[idx] = [old_x, old_y]
            continue
            
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / temp):
                current_score = new_score
            else:
                current_points[idx] = [old_x, old_y]
                
        temp *= cooling
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    global_best_points = generate_grid_config()
    global_best_score, _ = count_4_lines_opt(global_best_points)
    
    # Multi-start
    num_restarts = 20
    per_restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * per_restart_time
        if time.time() > restart_deadline - 0.2:
            break
        
        # Choose initial config
        if restart == 0:
            init_pts = generate_grid_config()
        elif restart == 1:
            init_pts = generate_perturbed_grid(scale=5)
        elif restart == 2:
            init_pts = generate_perturbed_grid(scale=10)
        elif restart == 3:
            init_pts = generate_perturbed_grid(scale=20)
        elif restart == 4:
            init_pts = generate_random_config(scale=100)
        elif restart == 5:
            init_pts = generate_random_config(scale=300)
        else:
            # Mix
            if random.random() < 0.5:
                init_pts = generate_perturbed_grid(scale=random.randint(5, 50))
            else:
                init_pts = generate_random_config(scale=random.randint(100, 500))
        
        best_pts, best_score = simulated_annealing(init_pts, restart_deadline)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_points = best_pts
            
    return global_best_points
# EVOLVE-BLOCK-END
