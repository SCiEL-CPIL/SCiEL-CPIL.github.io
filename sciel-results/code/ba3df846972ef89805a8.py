import random
import time
from collections import defaultdict
from math import gcd

def count_exact_four_lines(points):
    """Count lines containing exactly 4 of the given points.
    Returns (score, five_count) where score is # of 4-point lines,
    five_count is # of lines with 5+ points (should be 0 for valid)."""
    n = len(points)
    line_counts = defaultdict(int)
    # For each pair of points, compute the line they define
    # Use normalized (dx, dy, c) where c = -(a*x0 + b*y0) with gcd normalization
    # Actually, use the line in form: (y2-y1)*x + (x1-x2)*y + (x2-x1)*y1 + (y1-y2)*x2 = 0
    # Normalize by gcd
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line: dy*x - dx*y + (dx*y1 - dy*x1) = 0
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            # Normalize
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g == 0:
                g = 1
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            else:
                pass
            a //= g
            b //= g
            c //= g
            line_counts[(a, b, c)] += 1
    
    four_count = 0
    five_count = 0
    for count in line_counts.values():
        if count == 4:
            four_count += 1
        elif count >= 5:
            five_count += 1
    return four_count, five_count

def evaluate(points):
    """Score: maximize 4-point lines, heavily penalize 5+ collinear."""
    four, five = count_exact_four_lines(points)
    # Heavy penalty for any 5+ collinear
    return four - 100 * five

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_points = None
    best_score = -1
    
    # Try multiple starting configurations
    random.seed(42)
    
    starting_configs = []
    
    # 1. 4x4 grid
    starting_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. 4x4 grid shifted
    starting_configs.append([[x*3, y*3] for x in range(4) for y in range(4)])
    
    # 3. Random points in a box
    for trial in range(20):
        pts = []
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        starting_configs.append(pts)
    
    # 4. Try some structured patterns
    # Points on a circle-like pattern
    import math
    for r in [20, 30, 40]:
        pts = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = round(r * math.cos(angle))
            y = round(r * math.sin(angle))
            p = [x, y]
            if p not in pts:
                pts.append(p)
            else:
                pts.append([x+1, y])
        # Ensure 16 distinct
        while len(pts) < 16:
            pts.append([random.randint(-100,100), random.randint(-100,100)])
        pts = pts[:16]
        # Ensure distinct
        seen = set()
        unique_pts = []
        for p in pts:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_pts.append(p)
        while len(unique_pts) < 16:
            unique_pts.append([random.randint(-100,100), random.randint(-100,100)])
        starting_configs.append(unique_pts[:16])
    
    for start_idx, init_pts in enumerate(starting_configs):
        if time.time() > deadline - 10:
            break
        
        # Simulated annealing
        current_pts = [list(p) for p in init_pts]
        current_score = evaluate(current_pts)
        best_local = [list(p) for p in current_pts]
        best_local_score = current_score
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < deadline - 5:
            if T < T_min:
                break
            
            # Pick a random point to move
            i = random.randint(0, 15)
            old_x, old_y = current_pts[i]
            
            # New position: random nearby or random anywhere
            if random.random() < 0.7:
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)
            else:
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            
            # Clamp
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            # Check distinctness
            if any(current_pts[j][0] == new_x and current_pts[j][1] == new_y for j in range(16) if j != i):
                T *= cooling
                continue
            
            current_pts[i] = [new_x, new_y]
            new_score = evaluate(current_pts)
            
            if new_score >= current_score or random.random() < pow(2.718281828, (new_score - current_score) / T):
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local = [list(p) for p in current_pts]
            else:
                current_pts[i] = [old_x, old_y]
            
            T *= cooling
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = [list(p) for p in best_local]
    
    if best_points is None:
        # Fallback: 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Final validation
    four, five = count_exact_four_lines(best_points)
    if five > 0:
        # Fallback to 4x4 grid if invalid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
