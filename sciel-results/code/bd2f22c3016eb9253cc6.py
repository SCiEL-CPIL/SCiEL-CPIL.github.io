import time
import random
from math import gcd

DEADLINE = 110  # Safety margin for 120s limit

def _line_key(p1, p2):
    """Get a canonical representation of the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return ('v', x1)
    elif dy == 0:
        return ('h', y1)
    else:
        g = gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0:
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _build_lines(points):
    """Build line -> set of point indices. Single pass."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _check_no_four_collinear(lines):
    """Check no line has 4+ points."""
    for pts in lines.values():
        if len(pts) >= 4:
            return False
    return True

def _is_valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 100 or abs(y) > 100:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    lines = _build_lines(points)
    return _check_no_four_collinear(lines)

def _gen_perturbed_grid(seed):
    """Generate points on a perturbed grid structure."""
    rng = random.Random(seed)
    base_xs = [-40, -20, 0, 20, 40]
    base_ys = [-30, -10, 10, 30]
    
    points = []
    for x in base_xs:
        for y in base_ys:
            px = x + rng.randint(-2, 2)
            py = y + rng.randint(-2, 2)
            px = max(-100, min(100, px))
            py = max(-100, min(100, py))
            points.append([px, py])
    
    # Ensure distinct
    seen = set()
    unique = []
    for p in points:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique.append(p)
    
    # If duplicates occurred, fill with random
    while len(unique) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            unique.append([x, y])
            
    return unique[:20]

def construct():
    start_time = time.time()
    
    # Try to find a valid configuration from a perturbed grid
    # This structure tends to have many collinear triples, lowering the score
    for seed in range(1000):
        if time.time() - start_time > DEADLINE:
            break
        pts = _gen_perturbed_grid(seed)
        if _is_valid(pts):
            return [list(p) for p in pts]
            
    # Fallback: Random points
    rng = random.Random(42)
    while True:
        if time.time() - start_time > DEADLINE:
             # Return last valid or empty? Must return 20 points.
             # If we haven't found one, return a simple valid set (e.g. parabola)
             # Parabola y=x^2 for x in -10..9 is valid (no 3 collinear? 
             # Actually points on a parabola are in general position, so no 3 collinear.
             # This is valid (no 4 collinear). Score will be high, but it's valid.
             return [[x, x*x] for x in range(-10, 10)]
        
        points = []
        while len(points) < 20:
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if (x, y) not in [tuple(p) for p in points]:
                points.append([x, y])
        if _is_valid(points):
            return [list(p) for p in points]
