import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Ensure unique representation for line direction
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy

            # Line equation: a*x + b*y + c = 0
            # where a = dy, b = -dx, c = -(a*x1 + b*y1)
            a = dy
            b = -dx
            c = -(a * p1[0] + b * p1[1])
            
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)

    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def _generate_seeds():
    """
    Generates a list of initial configurations to start the search.
    """
    seeds = []
    
    # 1. Standard Grids
    # 3x4 Grid
    seeds.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 Grid
    seeds.append([[x, y] for x in range(3) for y in range(4)])
    # 2x6 Grid
    seeds.append([[x, y] for x in range(6) for y in range(2)])
    # 6x2 Grid
    seeds.append([[x, y] for x in range(2) for y in range(6)])
    
    # 2. Diagonal Shifts (Sheared Grids)
    # y in 0..3, x in 0..2, point (x+y, y)
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    seeds.append(pts_diag)
    
    # 3. Hesse-like / High Collinearity Structures
    # Try to construct configurations with many 3-point lines
    # Example: 4 points on y=0, 4 on y=1, 4 on y=2 with shifts
    # We'll generate a few random sheared grids
    rng_seed = random.Random(123)
    for _ in range(5):
        shift = rng_seed.randint(-2, 2)
        pts = []
        for y in range(3):
            for x in range(4):
                pts.append([x + y * shift, y])
        # Ensure distinct
        if len(set(map(tuple, pts))) == 12:
             seeds.append(pts)

    # 4. Random Dense Clusters
    for _ in range(10):
        pts = []
        for _ in range(12):
            pts.append([rng_seed.randint(-10, 10), rng_seed.randint(-10, 10)])
        if len(set(map(tuple, pts))) == 12:
            # Check not all collinear
            (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
            all_col = True
            for i in range(3, 12):
                xi, yi = pts[i]
                if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                    all_col = False
                    break
            if not all_col:
                seeds.append(pts)

    return seeds

def _find_collinear_candidates(points):
    """
    Finds potential integer points that lie on the line defined by two existing points.
    Returns a list of (index_to_move, new_x, new_y, line_key)
    """
    candidates = []
    n = len(points)
    occupied = set(map(tuple, points))
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            
            # Generate points on this line: p1 + k*(dx, dy)
            # We look for k values that result in integer coordinates within bounds
            # and are not already occupied.
            # Since dx, dy are coprime, any integer k works for integrality.
            # We check k in a reasonable range around the existing points.
            
            # Determine range for k based on bounds
            # |p1[0] + k*dx| <= 10^9
            # |p1[1] + k*dy| <= 10^9
            
            # Simple approach: try k from -20 to 20 relative to p1's index in the line?
            # Better: just try a fixed set of offsets from p1 and p2.
            for k in range(-10, 11):
                # Skip k=0 (p1) and k=1 (p2) if we just want new points?
                # Actually, we want to move a point *to* this position.
                # The position is (p1[0] + k*dx, p1[1] + k*dy).
                nx = p1[0] + k * dx
                ny = p1[1] + k * dy
                
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                    
                if (nx, ny) in occupied:
                    continue
                    
                # This is a valid candidate target.
                # We don't know which point to move yet, but we know the target.
                # We will store the target.
                candidates.append((nx, ny))
                
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    seeds = _generate_seeds()
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate seeds
    for seed in seeds:
        score = count_ordinary_lines(seed)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in seed]
        if best_score == 6:
            break
            
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    rng = random.Random(42)
    
    # Local Search
    while time.time() < deadline:
        # Strategy 1: Move a point to a collinear target (High probability)
        if rng.random() < 0.6:
            targets = _find_collinear_candidates(best_points)
            if not targets:
                continue
                
            target = rng.choice(targets)
            tx, ty = target
            
            # Pick a random point to move
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            if old_p == [tx, ty]:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = [tx, ty]
            
            # Check validity
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break
                    
        # Strategy 2: Random Perturbation (Medium probability)
        elif rng.random() < 0.9:
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]
            
            # Random step, allow larger jumps
            step_size = rng.choice([1, 2, 5, 10, 20, 50])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            
            if dx == 0 and dy == 0:
                continue
                
            new_p = [old_p[0] + dx, old_p[1] + dy]
            
            if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break
                    
        # Strategy 3: Swap two points (Low probability)
        else:
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i][:]
            p2 = best_points[j][:]
            
            # Perturb both
            s1 = rng.choice([1, 2, 5])
            s2 = rng.choice([1, 2, 5])
            
            dx1 = rng.randint(-s1, s1)
            dy1 = rng.randint(-s1, s1)
            dx2 = rng.randint(-s2, s2)
            dy2 = rng.randint(-s2, s2)
            
            new_p1 = [p1[0] + dx1, p1[1] + dy1]
            new_p2 = [p2[0] + dx2, p2[1] + dy2]
            
            if abs(new_p1[0]) > 10**9 or abs(new_p1[1]) > 10**9:
                continue
            if abs(new_p2[0]) > 10**9 or abs(new_p2[1]) > 10**9:
                continue
                
            new_points = [p[:] for p in best_points]
            new_points[i] = new_p1
            new_points[j] = new_p2
            
            if len(set(map(tuple, new_points))) < 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                if best_score == 6:
                    break

    return best_points
