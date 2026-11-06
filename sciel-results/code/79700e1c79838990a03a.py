"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 1, x1)  # vertical line: x = x1
    if dy == 0:
        return (1, 0, y1)  # horizontal line: y = y1
    # Normalize: make dx positive, or if dx==0 make dy positive
    if dx < 0:
        dx = -dx
        dy = -dy
    # Reduce by gcd
    g = abs(dy)
    while g:
        dx, g = g, dx % g
    if g == 0:
        # This shouldn't happen for distinct points
        return (0, 1, x1)
    # Now dx is the gcd of original dx,dy (absolute)
    # Actually let's do it properly
    import math
    g = math.gcd(abs(x2-x1), abs(y2-y1))
    dx = (x2-x1)//g
    dy = (y2-y1)//g
    # Canonical direction: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    # Map each line to the set of points on it
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
    
    count = 0
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
    return count

def _score(points):
    """Score: number of exact-4 lines, minus penalty if any 5+ collinear."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
    
    count = 0
    has_five = False
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
        elif len(pts) >= 5:
            has_five = True
    
    if has_five:
        return -10000 + count  # Heavy penalty
    return count

def _valid(points):
    """Check distinctness and bounds."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    return True

def _try_affine_transform(base_points, a, b, c, d, tx, ty):
    """Apply affine transform: x' = a*x + b*y + tx, y' = c*x + d*y + ty"""
    pts = []
    for x, y in base_points:
        nx = a * x + b * y + tx
        ny = c * x + d * y + ty
        pts.append([nx, ny])
    return pts

def _random_affine(base_points):
    """Generate a random affine transform of base_points, checking bounds."""
    for _ in range(200):
        # Random 2x2 matrix with small integers
        a = random.randint(-3, 3)
        b = random.randint(-3, 3)
        c = random.randint(-3, 3)
        d = random.randint(-3, 3)
        # Check determinant non-zero
        det = a * d - b * c
        if det == 0:
            continue
        
        pts = []
        for x, y in base_points:
            nx = a * x + b * y
            ny = c * x + d * y
            pts.append((nx, ny))
        
        # Find bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        
        # Center and check if it fits
        tx = random.randint(-500, 500)
        ty = random.randint(-500, 500)
        
        all_ok = True
        for nx, ny in pts:
            nx2 = nx + tx
            ny2 = ny + ty
            if abs(nx2) > 1000 or abs(ny2) > 1000:
                all_ok = False
                break
        if not all_ok:
            continue
        
        result = [[nx + tx, ny + ty] for nx, ny in pts]
        if _valid(result):
            return result
    return None

def _local_search(points, time_limit=10.0):
    """Local search: move one point at a time to improve score."""
    best = [list(p) for p in points]
    best_score = _score(best)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each point to a nearby random location
        for i in range(16):
            if time.time() - start_time >= time_limit:
                break
            for _ in range(3):
                # Random perturbation
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                new_x = best[i][0] + dx
                new_y = best[i][1] + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                # Check distinctness
                trial = [list(p) for p in best]
                trial[i] = [new_x, new_y]
                if len(set(tuple(p) for p in trial)) != 16:
                    continue
                
                s = _score(trial)
                if s > best_score:
                    best = trial
                    best_score = s
                    improved = True
                    break
        if not improved:
            # Try larger moves
            for i in range(16):
                if time.time() - start_time >= time_limit:
                    break
                for _ in range(5):
                    dx = random.randint(-20, 20)
                    dy = random.randint(-20, 20)
                    if dx == 0 and dy == 0:
                        continue
                    new_x = best[i][0] + dx
                    new_y = best[i][1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    trial = [list(p) for p in best]
                    trial[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) != 16:
                        continue
                    
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        break
            if not improved:
                # Stuck, try a random restart of one point
                i = random.randint(0, 15)
                trial = [list(p) for p in best]
                trial[i] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                if len(set(tuple(p) for p in trial)) == 16:
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                    # Even if not improved, sometimes accept to escape local opt
                    elif random.random() < 0.1:
                        best = trial
                        best_score = s
                if time.time() - start_time >= time_limit:
                    break
    
    return best, best_score

def _grid_based_search(time_limit=20.0):
    """Search over structured grid-like configurations."""
    best_pts = None
    best_score = -1
    start_time = time.time()
    
    # Try various grid sizes and spacings
    for size in range(3, 8):
        if time.time() - start_time >= time_limit:
            break
        for sx in range(1, 50):
            if time.time() - start_time >= time_limit:
                break
            for sy in range(1, 50):
                if time.time() - start_time >= time_limit:
                    break
                # Generate grid points
                pts = []
                for i in range(size):
                    for j in range(size):
                        x = i * sx
                        y = j * sy
                        pts.append([x, y])
                if len(pts) != 16:
                    continue
                
                # Check bounds
                if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                    continue
                
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with the 4x4 grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in base_grid]
    best_score = _score(best_pts)
    
    # Try affine transforms of the base grid
    for _ in range(500):
        if time.time() > deadline:
            break
        transformed = _random_affine(base_grid)
        if transformed is not None:
            s = _score(transformed)
            if s > best_score:
                best_score = s
                best_pts = transformed
    
    # Local search from best
    if time.time() < deadline - 5:
        pts, s = _local_search(best_pts, time_limit=min(15.0, deadline - time.time() - 2))
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Grid-based search
    if time.time() < deadline - 5:
        pts, s = _grid_based_search(time_limit=min(15.0, deadline - time.time() - 2))
        if pts is not None and s > best_score:
            best_score = s
            best_pts = pts
    
    # More local search from the best found
    if time.time() < deadline - 3:
        pts, s = _local_search(best_pts, time_limit=min(10.0, deadline - time.time() - 1))
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Final check
    if not _valid(best_pts):
        # Fallback to original grid
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts

# EVOLVE-BLOCK-END
