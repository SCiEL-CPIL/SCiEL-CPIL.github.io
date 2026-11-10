"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_exactly_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map from normalized line key to set of point indices
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            if dx == 0:
                # vertical line: key = ('v', x)
                key = ('v', x1)
            elif dy == 0:
                # horizontal line: key = ('h', y)
                key = ('h', y1)
            else:
                # General line: normalize (dx, dy) and compute intercept
                # Line: y = (dy/dx)x + b  =>  dy*x - dx*y = dy*x1 - dx*y1
                # Use normalized form: divide by gcd
                g = abs(_gcd(dx, dy))
                ndx, ndy = dx // g, dy // g
                # Ensure consistent sign: make ndx > 0, or if ndx == 0 then ndy > 0
                if ndx < 0 or (ndx == 0 and ndy < 0):
                    ndx, ndy = -ndx, -ndy
                # Intercept: ndy * x1 - ndx * y1
                c = ndy * x1 - ndx * y1
                key = ('g', ndx, ndy, c)
            
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    for key, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _get_lines_with_count(points):
    """Return dict mapping line key to set of indices, for all lines with >= 2 points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                key = ('v', x1)
            elif dy == 0:
                key = ('h', y1)
            else:
                g = abs(_gcd(dx, dy))
                ndx, ndy = dx // g, dy // g
                if ndx < 0 or (ndx == 0 and ndy < 0):
                    ndx, ndy = -ndx, -ndy
                c = ndy * x1 - ndx * y1
                key = ('g', ndx, ndy, c)
            
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def _count_exactly_four_from_map(line_map):
    count = 0
    for key, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any line contains 5 or more points."""
    line_map = _get_lines_with_count(points)
    for idx_set in line_map.values():
        if len(idx_set) >= 5:
            return True
    return False

def _valid(points):
    """Check distinctness, bounds, and no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return not _has_five_collinear(points)

def _random_points_in_box(bound=100):
    """Generate 16 random distinct integer points within [-bound, bound]."""
    while True:
        pts = []
        seen = set()
        for _ in range(16):
            x = random.randint(-bound, bound)
            y = random.randint(-bound, bound)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        if len(pts) == 16 and not _has_five_collinear(pts):
            return pts

def _hill_climb(points, time_limit, rng):
    """Local search: try moving each point to a nearby location."""
    best = [p[:] for p in points]
    best_score = _count_exactly_four_lines(best)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        # Try each point, try moving it
        for i in range(16):
            if time.time() - start > time_limit:
                break
            x, y = best[i]
            # Try random nearby moves
            for _ in range(50):
                nx = x + rng.randint(-5, 5)
                ny = y + rng.randint(-5, 5)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in [(p[0], p[1]) for j, p in enumerate(best) if j != i]:
                    continue
                trial = [p[:] for p in best]
                trial[i] = [nx, ny]
                if _has_five_collinear(trial):
                    continue
                score = _count_exactly_four_lines(trial)
                if score > best_score:
                    best = trial
                    best_score = score
                    improved = True
                    break  # restart loop for this point
        if not improved:
            # Try larger moves
            for i in range(16):
                if time.time() - start > time_limit:
                    break
                x, y = best[i]
                for _ in range(20):
                    nx = x + rng.randint(-20, 20)
                    ny = y + rng.randint(-20, 20)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if (nx, ny) in [(p[0], p[1]) for j, p in enumerate(best) if j != i]:
                        continue
                    trial = [p[:] for p in best]
                    trial[i] = [nx, ny]
                    if _has_five_collinear(trial):
                        continue
                    score = _count_exactly_four_lines(trial)
                    if score > best_score:
                        best = trial
                        best_score = score
                        improved = True
                        break
            if not improved:
                break
    return best, best_score

def _simulated_annealing(points, time_limit, rng):
    """SA with diverse moves."""
    best = [p[:] for p in points]
    best_score = _count_exactly_four_lines(best)
    current = [p[:] for p in best]
    current_score = best_score
    start = time.time()
    T = 5.0
    cooling = 0.999
    
    while time.time() - start < time_limit:
        # Choose a point to move
        i = rng.randint(0, 15)
        x, y = current[i]
        step = int(T * 3) + 1
        nx = x + rng.randint(-step, step)
        ny = y + rng.randint(-step, step)
        if abs(nx) > 1000 or abs(ny) > 1000:
            T *= cooling
            continue
        if (nx, ny) in [(p[0], p[1]) for j, p in enumerate(current) if j != i]:
            T *= cooling
            continue
        
        trial = [p[:] for p in current]
        trial[i] = [nx, ny]
        if _has_five_collinear(trial):
            T *= cooling
            continue
        
        score = _count_exactly_four_lines(trial)
        delta = score - current_score
        if delta >= 0 or rng.random() < min(1.0, delta / max(T, 0.1)):
            current = trial
            current_score = score
            if current_score > best_score:
                best = [p[:] for p in current]
                best_score = current_score
        T *= cooling
    
    return best, best_score

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    rng = random.Random(42)
    
    best_points = None
    best_score = 0
    
    # Start with the 4x4 grid as a known baseline
    grid = [[x, y] for x in range(4) for y in range(4)]
    if _valid(grid):
        grid_score = _count_exactly_four_lines(grid)
        if grid_score > best_score:
            best_score = grid_score
            best_points = [p[:] for p in grid]
    
    # Try multiple random starts with hill climbing
    while time.time() < deadline - 20:
        # Random start
        pts = _random_points_in_box(50)
        # Hill climb from this start
        hc_pts, hc_score = _hill_climb(pts, min(5, deadline - time.time() - 20), rng)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts
    
    # Simulated annealing from best found so far
    if best_points is not None and time.time() < deadline - 10:
        sa_pts, sa_score = _simulated_annealing(best_points, min(30, deadline - time.time() - 10), rng)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Final hill climb polish
    if best_points is not None and time.time() < deadline - 2:
        final_pts, final_score = _hill_climb(best_points, min(10, deadline - time.time() - 2), rng)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
