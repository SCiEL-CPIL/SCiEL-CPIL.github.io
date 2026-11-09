"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _count_ordinary_lines(pts):
    """Count lines containing exactly two of the points."""
    n = len(pts)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                key = (0, 0, pts[i][0])
            elif dy == 0:
                key = (0, 1, pts[i][1])
            else:
                if dy < 0:
                    dx, dy = -dx, -dy
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                # Normal line form: dy*x - dx*y = c
                c = dy * pts[i][0] - dx * pts[i][1]
                key = (1, dx, dy, c)
            lines[key] = lines.get(key, 0) + 1
    return sum(1 for c in lines.values() if c == 1)

def _evaluate(points):
    """Score: number of ordinary lines (lower is better)."""
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    for j in range(1, n):
        for k in range(j + 1, n):
            x0, y0 = points[0]
            x1, y1 = points[j]
            x2, y2 = points[k]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                return True
    return False

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    # Start with a 3x4 grid configuration
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _evaluate(best_points)
    
    if not _not_all_collinear(best_points):
        # Fallback: add a point off the grid
        best_points[0] = [0, 0]
        best_points[1] = [1, 1]
        best_points[2] = [2, 4]
        best_score = _evaluate(best_points)
    
    # Hill climbing: try to improve by moving random points
    n = 12
    rng = random.Random(42)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(n)
        old_x, old_y = best_points[idx]
        
        # Try a random new position
        new_x = rng.randint(-50, 50)
        new_y = rng.randint(-50, 50)
        
        # Skip if same point
        if new_x == old_x and new_y == old_y:
            continue
        
        # Check for duplicate points
        is_dup = False
        for i, (px, py) in enumerate(best_points):
            if i != idx and px == new_x and py == new_y:
                is_dup = True
                break
        if is_dup:
            continue
        
        # Tentatively move the point
        best_points[idx] = [new_x, new_y]
        
        if not _not_all_collinear(best_points):
            best_points[idx] = [old_x, old_y]
            continue
        
        score = _evaluate(best_points)
        
        if score < best_score:
            best_score = score
            # Keep the move
        else:
            # Revert
            best_points[idx] = [old_x, old_y]
        
        # If we've found a very good score, continue searching
        if best_score <= 6:
            break
    
    # Final validation
    assert len(best_points) == 12
    assert all(isinstance(p, list) and len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert _not_all_collinear(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
