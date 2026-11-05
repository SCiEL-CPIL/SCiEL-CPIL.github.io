"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Deadline guard: evaluator kills after 150s wall-clock. Use 130s as safety margin.
_DEADLINE = 130.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use a dict to map line (normalized) -> count of points on it
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            # Ensure canonical form: make the first non-zero component positive
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Now we need a unique key for the line.
            # The line is determined by direction (dx, dy) and a point on it.
            # We can use the line equation: dy*x - dx*y = c
            # where c = dy*x1 - dx*y1
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    # Count lines with exactly 2 points
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _score(points):
    """Lower is better: number of ordinary lines."""
    return _count_ordinary_lines(points)

def _is_valid(points):
    """Check that points are 12 distinct integer points with |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        x1, y1 = points[0]
        x2, y2 = points[1]
        for i in range(2, 12):
            x3, y3 = points[i]
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross != 0:
                return True
        return False  # All collinear
    return True

def _known_configurations():
    """Return a list of known good starting configurations for 12 points."""
    configs = []
    
    # 3x4 grid (baseline)
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    configs.append(grid_3x4)
    
    # 4x3 grid
    grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
    configs.append(grid_4x3)
    
    # Try a configuration based on finite field / combinatorial designs
    # The Hessian configuration has 9 points, we can extend it.
    # Hessian configuration: 9 points, 12 lines, each line has 3 points, each point on 4 lines.
    # For 12 points, we can try to find a configuration with few ordinary lines.
    
    # A known construction: take points from a 2x6 grid but perturb some
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    configs.append(grid_2x6)
    
    # 6x2 grid
    grid_6x2 = [[x, y] for x in range(2) for y in range(6)]
    configs.append(grid_6x2)
    
    # Try a configuration with more collinearities
    # e.g., 3 points on each of 4 lines, with some sharing
    # This is hard to construct by hand, so we'll rely on search.
    
    # A configuration inspired by the Sylvester-Gallai theorem bounds
    # For 12 points, minimum ordinary lines over reals is 6.
    # Try to construct something close to that.
    
    # Let's try a "star" configuration: 1 center point, 11 others arranged so many lines pass through center
    # But that gives many lines through center with 2 points (center + 1 other), which is bad.
    
    # Better: arrange points so that many lines have 3+ points.
    # A 3x4 grid has: 
    # - 4 vertical lines with 3 points each (not ordinary)
    # - 3 horizontal lines with 4 points each (not ordinary)
    # - Diagonals: some have 2 points (ordinary), some have 3
    # Let's count for 3x4 grid:
    # Points: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)
    # Vertical lines x=0,1,2,3: 4 lines, each with 3 points -> not ordinary
    # Horizontal lines y=0,1,2: 3 lines, each with 4 points -> not ordinary
    # Diagonals with slope 1: (0,0)-(1,1)-(2,2), (1,0)-(2,1)-(3,2), (0,1)-(1,2), (2,0)-(3,1)
    #   First two have 3 points, last two have 2 points -> 2 ordinary
    # Diagonals with slope -1: (0,2)-(1,1)-(2,0), (1,2)-(2,1)-(3,0), (0,1)-(1,0), (2,2)-(3,1)
    #   First two have 3 points, last two have 2 points -> 2 ordinary
    # Other diagonals: slope 2, -2, 1/2, -1/2, etc.
    # This is getting complex. Let's just compute it.
    
    return configs

def _local_search(points, max_iters=10000):
    """Improve a configuration via local search (swap a point to a random nearby location)."""
    points = [p[:] for p in points]
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        if iteration % 100 == 0:
            temp *= cooling
        
        # Pick a random point and move it
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try a random new position in a reasonable range
        # Keep coordinates small to avoid overflow and maintain integer precision
        new_x = random.randint(-20, 20)
        new_y = random.randint(-20, 20)
        
        # Ensure distinctness
        new_point = (new_x, new_y)
        if any(new_point == (points[i][0], points[i][1]) for i in range(12) if i != idx):
            continue
        
        points[idx] = [new_x, new_y]
        new_score = _score(points)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.718281828, -delta / max(temp, 0.1)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            points[idx] = [old_x, old_y]
    
    return best_points, best_score

def _construct():
    """Main construction function with time management."""
    random.seed(42)
    
    # Get initial configurations
    initial_configs = _known_configurations()
    
    best_points = None
    best_score = float('inf')
    
    # First, evaluate all initial configurations
    for config in initial_configs:
        if not _is_valid(config):
            continue
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in config]
    
    # If no valid config found, use the 3x4 grid as fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _score(best_points)
    
    # Local search from each initial configuration
    for config in initial_configs:
        if _time_left() < 10.0:
            break
        if not _is_valid(config):
            continue
        improved_points, improved_score = _local_search(config, max_iters=5000)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_points
    
    # If we still have time, do more random restarts
    while _time_left() > 15.0:
        # Generate a random configuration
        random_points = []
        for _ in range(12):
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in [(p[0], p[1]) for p in random_points]:
                random_points.append([x, y])
        
        if len(random_points) == 12 and _is_valid(random_points):
            improved_points, improved_score = _local_search(random_points, max_iters=3000)
            if improved_score < best_score:
                best_score = improved_score
                best_points = improved_points
    
    # Final validation
    if not _is_valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

def construct():
    """Entry point: returns exactly 12 distinct integer points."""
    global _START_TIME
    _START_TIME = time.time()
    return _construct()
# EVOLVE-BLOCK-END
