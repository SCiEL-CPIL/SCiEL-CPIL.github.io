import time
import random
from math import gcd
from itertools import combinations

def count_exact_four_lines_fast(points):
    """Count lines containing exactly 4 of the 16 points. Returns (four_count, five_or_more_count)."""
    n = len(points)
    if n < 4:
        return 0, 0
    
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    five_or_more = 0
    for pair_count in line_counts.values():
        if pair_count == 6:  # C(4,2) = 6
            four_count += 1
        elif pair_count >= 10:  # C(5,2) = 10
            five_or_more += 1
    
    return four_count, five_or_more

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_score = 0
    
    random.seed(42)
    
    # Initial configuration: 4x4 grid
    # This has 10 lines of exactly 4 points.
    initial_grid = [[x, y] for x in range(4) for y in range(4)]
    s, f = count_exact_four_lines_fast(initial_grid)
    if f == 0:
        best_score = s
        best_points = [p[:] for p in initial_grid]
    
    # List of configurations to try
    configs = [ [p[:] for p in best_points] ]
    
    # Add some sheared grids
    for shear in range(-3, 4):
        pts = []
        for i in range(4):
            for j in range(4):
                x = 4 * i + shear * j
                y = 4 * j
                pts.append([x, y])
        if len(set(tuple(p) for p in pts)) == 16:
            if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                configs.append(pts)
                
    # Evaluate initial configs
    for pts in configs:
        if time.time() > deadline:
            break
        s, f = count_exact_four_lines_fast(pts)
        if f == 0 and s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            
    current = [p[:] for p in best_points]
    current_score = best_score
    
    # Local Search with Swaps and Moves
    # We will alternate between single-point moves and swaps.
    
    temperature = 5.0
    cooling = 0.999
    no_improve_count = 0
    
    while time.time() < deadline:
        no_improve_count += 1
        
        if no_improve_count > 1000:
            # Restart
            current = [p[:] for p in best_points]
            current_score = best_score
            temperature = 5.0
            no_improve_count = 0
            # Perturb
            for _ in range(3):
                idx = random.randint(0, 15)
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_x = current[idx][0] + dx
                new_y = current[idx][1] + dy
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    # Check distinctness
                    tmp = current[idx]
                    current[idx] = [new_x, new_y]
                    if len(set(tuple(p) for p in current)) == 16:
                        break
                    else:
                        current[idx] = tmp
            
        # Choose move type: 50% Swap, 50% Move
        if random.random() < 0.5:
            # Swap two points
            i, j = random.sample(range(16), 2)
            new_pts = [p[:] for p in current]
            new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
            
            # Validation: distinctness is preserved in swap. Bounds are preserved.
            s, f = count_exact_four_lines_fast(new_pts)
            
            if f > 0:
                delta = s - current_score - 100 * f
            else:
                delta = s - current_score
                
            if delta > 0 or random.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                current = new_pts
                current_score = s
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
                    no_improve_count = 0
                    temperature = 5.0
                else:
                    temperature *= cooling
        else:
            # Move one point
            idx = random.randint(0, 15)
            # Try to move to a "good" spot?
            # For now, random small move
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_pts = [p[:] for p in current]
            new_pts[idx] = [current[idx][0] + dx, current[idx][1] + dy]
            
            if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
                
            s, f = count_exact_four_lines_fast(new_pts)
            
            if f > 0:
                delta = s - current_score - 100 * f
            else:
                delta = s - current_score
                
            if delta > 0 or random.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                current = new_pts
                current_score = s
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
                    no_improve_count = 0
                    temperature = 5.0
                else:
                    temperature *= cooling

    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16, "Points not distinct"
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final), "Points out of bounds"
    s, f = count_exact_four_lines_fast(final)
    assert f == 0, f"Five or more collinear: {f}"
    
    return final
