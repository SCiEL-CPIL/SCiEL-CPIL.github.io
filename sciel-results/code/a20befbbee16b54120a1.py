import random
import time
from math import gcd, exp

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _build_line_index(points):
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_score = -1
    
    # Run multiple trials with different initial configurations and annealing schedules
    for trial in range(50): # 50 trials
        if time.time() > deadline - 10:
            break
            
        # Initial configuration
        if trial == 0:
            # 4x4 grid
            points = [[x, y] for x in range(4) for y in range(4)]
        elif trial == 1:
            # 4x4 grid shifted
            points = [[x, y] for x in range(4) for y in range(4)]
            points[0][0] += 1
        else:
            # Random grid-like
            points = []
            for i in range(4):
                for j in range(4):
                    points.append([i + random.randint(-2, 2), j + random.randint(-2, 2)])
            # Ensure distinct
            # Simple check, if collision, retry or adjust
            # For simplicity, just use random points in a small box for other trials
            # Actually, let's use a perturbed grid for all
            points = [[x + random.randint(-3, 3), y + random.randint(-3, 3)] for x in range(4) for y in range(4)]
            
        # Check distinctness and bounds
        valid_init = True
        seen = set()
        for p in points:
            if p[0] in seen or p[1] in seen: # Wait, seen should be (x,y)
                pass
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                valid_init = False
                break
            if (p[0], p[1]) in seen:
                valid_init = False
                break
            seen.add((p[0], p[1]))
        
        if not valid_init:
            continue
            
        line_points = _build_line_index(points)
        is_valid, current_score = _evaluate_from_index(points, line_points)
        
        if not is_valid:
            # Try to fix invalidity by moving points
            # Or just skip this trial if it's bad
            pass
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 5 and T > min_T:
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Generate move
            # Bias towards small moves
            r = random.random()
            if r < 0.5:
                dx = random.randint(-1, 1)
                dy = random.randint(-1, 1)
            elif r < 0.8:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
            else:
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = [new_x, new_y]
            
            # Check distinctness
            is_distinct = True
            for i, p in enumerate(points):
                if i != idx:
                    if p[0] == new_x and p[1] == new_y:
                        is_distinct = False
                        break
            if not is_distinct:
                continue
                
            # Apply
            points[idx] = new_point
            new_line_points = _build_line_index(points) # Rebuild for simplicity, 16 points is fast
            is_valid, new_score = _evaluate_from_index(points, new_line_points)
            
            # Acceptance
            if is_valid:
                if new_score > current_score:
                    accept = True
                elif new_score == current_score:
                    accept = random.random() < 0.5
                else:
                    # Worse
                    delta = new_score - current_score
                    accept_prob = exp(delta / T)
                    accept = random.random() < accept_prob
            else:
                # Invalid, reject unless we want to explore invalid space (risky)
                accept = False
                
            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
            else:
                # Revert
                points[idx] = [old_x, old_y]
                
            T *= cooling
            
        # End of trial
        if best_score > 15:
            break # Found something very good
            
    if best_points is None:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
