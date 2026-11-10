import time
import random
from math import gcd

def normalize_line(a, b, c):
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        return (0, 0, 0)
    a //= g
    b //= g
    c //= g
    # Canonical form: first non-zero coefficient is positive
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def evaluate(points):
    line_map = {}
    n = len(points)
    # Pre-calc pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            # A(x1-x2) + B(y1-y2) + C = 0 is not the standard form
            # Standard: A x + B y + C = 0
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            key = normalize_line(a, b, c)
            if key in line_map:
                line_map[key].add(i)
                line_map[key].add(j)
            else:
                line_map[key] = {i, j}
    
    count = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k >= 5:
            return 0, False
        if k == 4:
            count += 1
    return count, True

def construct():
    start_time = time.time()
    # 110s limit to be safe
    deadline = start_time + 110.0
    
    # Initial configuration: 4x4 grid
    # This has 10 lines of 4 points (4 rows, 4 cols, 2 diags)
    # and no lines of 5+ points.
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Local search
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Decide move type
        # 90% local, 10% global
        if random.random() < 0.1:
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
        else:
            # Small local perturbation
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
        if new_x == old_x and new_y == old_y:
            continue
            
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        # Check for duplicates
        is_dup = False
        for k, p in enumerate(current_points):
            if k != idx and p[0] == new_x and p[1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Apply move
        current_points[idx] = [new_x, new_y]
        
        # Evaluate
        score, valid = evaluate(current_points)
        
        if valid:
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in current_points]
                
            # Accept if better or equal (greedy)
            # To escape local minima, we could accept worse with some prob,
            # but restarts handle that.
            if score >= current_score:
                current_score = score
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
        else:
            # Revert
            current_points[idx] = [old_x, old_y]

    return best_points
