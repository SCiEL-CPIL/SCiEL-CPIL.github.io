"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _try_move(pts, idx, new_x, new_y):
    """Try moving point idx to (new_x, new_y), return new pts if valid."""
    if new_x == pts[idx][0] and new_y == pts[idx][1]:
        return None
    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
        return None
    # Check duplicate
    for i, p in enumerate(pts):
        if i != idx and p[0] == new_x and p[1] == new_y:
            return None
    new_pts = [list(p) for p in pts]
    new_pts[idx] = [new_x, new_y]
    return new_pts

def _generate_grid(w, h):
    """Generate a w x h grid of points."""
    return [[x, y] for x in range(w) for y in range(h)]

def _generate_ellipse(n_points):
    """Generate points on an ellipse x^2 + y^2 = R^2 scaled."""
    # Use Pythagorean triples to find integer points on a circle
    # (3,4,5), (5,12,13), (8,15,17), (7,24,25), (20,21,29), (12,35,37), (9,40,41), (28,45,53)
    triples = [
        (3, 4), (4, 3), (-3, 4), (-4, 3),
        (3, -4), (4, -3), (-3, -4), (-4, -3),
        (5, 12), (12, 5), (-5, 12), (-12, 5),
        (5, -12), (12, -5), (-5, -12), (-12, -5),
        (8, 15), (15, 8), (-8, 15), (-15, 8),
        (8, -15), (15, -8), (-8, -15), (-15, -8),
        (7, 24), (24, 7), (-7, 24), (-24, 7),
        (7, -24), (24, -7), (-7, -24), (-24, -7),
        (20, 21), (21, 20), (-20, 21), (-21, 20),
        (20, -21), (21, -21), (-20, -21), (-21, -20),
        (12, 35), (35, 12), (-12, 35), (-35, 12),
        (12, -35), (35, -12), (-12, -35), (-35, -12),
        (9, 40), (40, 9), (-9, 40), (-40, 9),
        (9, -40), (40, -9), (-9, -40), (-40, -9),
        (28, 45), (45, 28), (-28, 45), (-45, 28),
        (28, -45), (45, -28), (-28, -45), (-45, -28),
    ]
    # We need 12 distinct points. Pick 12 from the list, ensuring they are distinct.
    # To make them fit in a smaller range, we can scale or just pick a subset.
    # Let's pick 12 well-distributed points.
    # Indices: 0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27
    indices = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27]
    pts = [list(triples[i]) for i in indices]
    return pts

def _generate_hyperbola(n_points):
    """Generate points on hyperbola xy = k."""
    # Let k = 120. Divisors of 120: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120
    # Points (x, 120/x)
    divisors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24]
    pts = [[d, 120//d] for d in divisors]
    return pts

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    initial_configs = []
    
    # Config 1: 3x4 grid
    initial_configs.append(_generate_grid(4, 3))
    
    # Config 2: 2x6 grid
    initial_configs.append(_generate_grid(6, 2))
    
    # Config 3: Ellipse points
    initial_configs.append(_generate_ellipse(12))
    
    # Config 4: Hyperbola points
    initial_configs.append(_generate_hyperbola(12))
    
    # Config 5: 4x3 grid shifted
    initial_configs.append([[x+1, y+1] for x in range(4) for y in range(3)])
    
    # Find best initial config
    best_pts = None
    best_score = float('inf')
    for config in initial_configs:
        if not _is_valid(config):
            continue
        score = _count_ordinary(config)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in config]
    
    if best_pts is None:
        best_pts = _generate_grid(4, 3)
        best_score = _count_ordinary(best_pts)
    
    random.seed(42)
    
    # Local search: Targeted moves to improve collinearity
    # Strategy: 
    # 1. Identify ordinary lines (lines with exactly 2 points).
    # 2. For each point on an ordinary line, try to move it to a position 
    #    that lies on a line with 3+ points, or creates a new 3+ point line.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Get current line map to identify ordinary lines and high-order lines
        n = len(best_pts)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = best_pts[i]
                x2, y2 = best_pts[j]
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        
        ordinary_lines = [key for key, v in line_map.items() if len(v) == 2]
        
        # Try to improve by moving points on ordinary lines
        # We prioritize moving points that are on multiple ordinary lines
        point_ordinary_count = {i: 0 for i in range(n)}
        for key in ordinary_lines:
            for i in line_map[key]:
                point_ordinary_count[i] += 1
        
        # Sort points by number of ordinary lines they are on (descending)
        candidates = sorted(range(n), key=lambda i: point_ordinary_count[i], reverse=True)
        
        for idx in candidates:
            if time.time() > deadline:
                break
            
            orig_x, orig_y = best_pts[idx]
            
            # Generate candidate new positions
            # 1. Intersections of lines defined by other points
            # 2. Points on existing high-order lines
            # 3. Small random perturbations
            
            # Find all lines formed by other points
            other_lines = {}
            for i in range(n):
                if i == idx: continue
                for j in range(i + 1, n):
                    if j == idx: continue
                    x1, y1 = best_pts[i]
                    x2, y2 = best_pts[j]
                    a = y2 - y1
                    b = x1 - x2
                    c = x2 * y1 - x1 * y2
                    g = gcd(gcd(abs(a), abs(b)), abs(c))
                    if g > 0:
                        a //= g
                        b //= g
                        c //= g
                    if a < 0 or (a == 0 and b < 0):
                        a, b, c = -a, -b, -c
                    key = (a, b, c)
                    if key not in other_lines:
                        other_lines[key] = set()
                    other_lines[key].add(i)
                    other_lines[key].add(j)
            
            # Candidate 1: Move to a point on a line with 3+ other points
            for key, pts_on_line in other_lines.items():
                if len(pts_on_line) >= 2:
                    # Pick a point on this line that is not idx (idx is not on it)
                    # We want to place idx on this line.
                    # Find the line equation ax + by + c = 0
                    a, b, c = key
                    # Find an integer point on this line that is not already occupied
                    # We can try to find a point close to the original position
                    # Or just pick a random point on the line
                    # Since we need integer coords, we can parameterize
                    # ax + by = -c
                    # If gcd(a,b) divides -c, there are solutions.
                    # Let's just try a few random points on the line
                    # We can pick a random x in a range and solve for y
                    # y = (-ax - c) / b
                    # We need b != 0
                    if b != 0:
                        # Pick x such that y is integer
                        # x = x0 + k * (b/g)
                        # y = y0 - k * (a/g)
                        # We don't have x0, y0 easily. Let's just try a few random x
                        for _ in range(5):
                            x_new = random.randint(-100, 100)
                            # Check if y is integer
                            num = -a * x_new - c
                            if num % b == 0:
                                y_new = num // b
                                if abs(x_new) <= 10**9 and abs(y_new) <= 10**9:
                                    new_pts = _try_move(best_pts, idx, x_new, y_new)
                                    if new_pts:
                                        score = _count_ordinary(new_pts)
                                        if score < best_score:
                                            best_pts = new_pts
                                            best_score = score
                                            improved = True
                                            break
                    if improved:
                        break
            
            if improved:
                break
            
            # Candidate 2: Small random moves
            for _ in range(10):
                step = random.randint(1, 5)
                new_x = orig_x + random.randint(-step, step)
                new_y = orig_y + random.randint(-step, step)
                new_pts = _try_move(best_pts, idx, new_x, new_y)
                if new_pts:
                    score = _count_ordinary(new_pts)
                    if score < best_score:
                        best_pts = new_pts
                        best_score = score
                        improved = True
                        break
            
            if improved:
                break
        
        # If no improvement in targeted moves, do a random restart phase
        if not improved and time.time() < deadline:
             # Try a few random moves to escape local minima
             for _ in range(100):
                 if time.time() > deadline:
                     break
                 idx = random.randint(0, 11)
                 orig_x, orig_y = best_pts[idx]
                 step = random.randint(1, 20)
                 new_x = orig_x + random.randint(-step, step)
                 new_y = orig_y + random.randint(-step, step)
                 new_pts = _try_move(best_pts, idx, new_x, new_y)
                 if new_pts:
                     score = _count_ordinary(new_pts)
                     if score < best_score:
                         best_pts = new_pts
                         best_score = score
                         improved = True
                         break

    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
