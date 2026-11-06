import random
import time
from math import gcd

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            if x1 == x2:
                key = (1, 0, -x1)
            elif y1 == y2:
                key = (0, 1, -y1)
            else:
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                g = gcd(abs(A), gcd(abs(B), abs(C)))
                if g > 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # This shouldn't happen for distinct points
                    continue
                
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
            
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
            
    ordinary = 0
    for count in lines.values():
        if count == 1:
            ordinary += 1
            
    return ordinary

def is_valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) < 3:
        return True
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
        for i in range(3, len(points)):
            xi, yi = points[i]
            if (y2 - y1) * (xi - x1) != (y3 - y1) * (x2 - x1):
                return True
        return False
    return True

def local_search(points, max_time=10):
    start = time.time()
    current = [list(p) for p in points]
    current_score = count_ordinary_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    while time.time() - start < max_time:
        improved = False
        for i in range(12):
            if time.time() - start > max_time:
                break
            x, y = current[i]
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pts = [list(p) for p in current]
                    new_pts[i] = [nx, ny]
                    if not is_valid(new_pts):
                        continue
                    new_score = count_ordinary_lines(new_pts)
                    if new_score < current_score:
                        current = new_pts
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best = [list(p) for p in current]
        if not improved:
            # Stuck, try a random jump
            i = random.randint(0, 11)
            current[i] = [random.randint(-50, 50), random.randint(-50, 50)]
            if not is_valid(current):
                # Revert if invalid
                # Actually, just restart the loop iteration
                continue
            current_score = count_ordinary_lines(current)
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_score = float('inf')
    
    # Seed 1: 3x4 Grid
    pts_grid_34 = [(i, j) for i in range(4) for j in range(3)]
    # Seed 2: 4x3 Grid
    pts_grid_43 = [(i, j) for i in range(3) for j in range(4)]
    # Seed 3: Hexagon + Midpoints (approx integer)
    # Scale by 2 to avoid fractions for midpoints of (1,0) and (0,1)?
    # V: (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1)
    # M: (1.5, 0.5)... not integer.
    # Use a larger hexagon.
    # V: (4,0), (2,2), (-2,2), (-4,0), (-2,-2), (2,-2)
    # M: (3,1), (0,2), (-3,1), (-3,-1), (0,-2), (3,-1)
    pts_hex = [
        (4, 0), (2, 2), (-2, 2), (-4, 0), (-2, -2), (2, -2),
        (3, 1), (0, 2), (-3, 1), (-3, -1), (0, -2), (3, -1)
    ]
    # Seed 4: Random
    pts_rand = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(12)]
    while len(set(pts_rand)) < 12:
        pts_rand[random.randint(0, 11)] = (random.randint(-10, 10), random.randint(-10, 10))

    seeds = [pts_grid_34, pts_grid_43, pts_hex, pts_rand]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        if not is_valid(seed):
            continue
        score = count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in seed]
        
        # Run local search from this seed
        if time.time() < deadline - 5:
            improved_pts, improved_score = local_search(seed, max_time=10)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_pts
    
    # Final check
    if best_points is None:
        return [[i, 0] for i in range(11)] + [[0, 1]]
        
    return [[x, y] for x, y in best_points]
