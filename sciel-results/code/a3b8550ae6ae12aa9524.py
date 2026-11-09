"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Use a set of normalized lines: (A, B, C) where line is Ax + By + C = 0
    # Normalize so that (A, B) is in canonical form
    line_map = {}
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            # Line through (x1,y1) and (x2,y2)
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize: make gcd divide, and canonical sign
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            # Canonical: first non-zero of (A, B, C) should be positive
            if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in line_map:
                line_map[key] = {i, j}
            else:
                line_map[key].add(i)
                line_map[key].add(j)
    
    ordinary = 0
    for pts_on_line in line_map.values():
        if len(pts_on_line) == 2:
            ordinary += 1
    return ordinary

def _validate(points):
    """Check that points are valid: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        if p in seen:
            return False
        seen.add(tuple(p))
    # Not all collinear
    n = len(points)
    x0, y0 = points[0][0], points[0][1]
    x1, y1 = points[1][0], points[1][1]
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    for i in range(2, n):
        if A * points[i][0] + B * points[i][1] + C != 0:
            return True
    return False

def _try_parabola_config():
    """
    Try placing 11 points on a parabola y = a*x^2 + b*x + c and the 12th at a strategic location.
    We search over rational parameterizations to maximize 3+ point lines.
    """
    best_score = 10**9
    best_points = None
    deadline = time.time() + 100  # safety margin
    
    # For points on a parabola, 3 points are collinear iff the sum of their x-coordinates satisfies a condition.
    # Specifically, for y = x^2, three points with x-coordinates x1, x2, x3 are collinear iff x1+x2+x3 = 0
    # (this is a known property of the parabola y=x^2).
    # More generally for y = a*x^2, the condition is x1+x2+x3 = 0 if a != 0.
    
    # Strategy: pick 11 x-values on the parabola y = x^2 (a=1, b=0, c=0)
    # such that as many triples as possible sum to 0.
    # Then place the 12th point to create additional 3-point lines.
    
    # For y = x^2, if we pick x-values from a symmetric set, we get many collinear triples.
    # Example: {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
    # Triples summing to 0: many.
    
    # Let's try symmetric sets of 11 integers
    # If we use a symmetric set around 0, like {-k, ..., -1, 0, 1, ..., k-1} (11 elements)
    # or other structured sets.
    
    # Known result: for the 12-point minimum (6 ordinary lines), one construction uses
    # a regular 12-gon or similar. But for integer points, we need to find good configurations.
    
    # Let's try several candidate configurations:
    
    # Configuration 1: 11 points on y=x^2 with x from a set, 12th point off
    # Try x-values: symmetric sets
    candidate_x_sets = [
        list(range(-5, 6)),  # -5 to 5, 11 values
        list(range(-6, 5)),  # -6 to 4, 11 values
        list(range(-4, 7)),  # -4 to 6, 11 values
        # Other structured sets
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6],
        [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10],
    ]
    
    for x_vals in candidate_x_sets:
        if len(x_vals) != 11:
            continue
        # Points on y = x^2
        pts = [[x, x*x] for x in x_vals]
        
        # Now place 12th point. Try several positions.
        # The 12th point should be chosen to create additional 3-point lines.
        # For a point P = (px, py) and two points (x1, x1^2), (x2, x2^2) on the parabola,
        # P is on the line through them iff a certain condition holds.
        
        # Try: place 12th point at (0, y) for various y, or at other strategic positions.
        # Also try: the point that makes it a 12-point configuration with known properties.
        
        # Simple approach: try a grid of candidate 12th points
        for px in range(-10, 11):
            for py in range(-20, 21):
                pts12 = pts + [[px, py]]
                if not _validate(pts12):
                    continue
                score = _count_ordinary_lines(pts12)
                if score < best_score:
                    best_score = score
                    best_points = pts12[:]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Also try: 6 points on y=x^2 and 6 points on another line or curve
    # Configuration 2: Points on two lines
    # 6 points on x-axis, 6 points on y=x (or y=x+1)
    for offset in range(-5, 6):
        pts = [[i, 0] for i in range(6)] + [[i, i + offset] for i in range(6)]
        if not _validate(pts):
            continue
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
    
    # Configuration 3: Points on a 3x4 or 4x3 grid
    for h in range(-3, 4):
        for w in range(-3, 4):
            pts = []
            for i in range(4):
                for j in range(3):
                    pts.append([i + w, j + h])
            if not _validate(pts):
                continue
            score = _count_ordinary_lines(pts)
            if score < best_score:
                best_score = score
                best_points = pts[:]
    
    # Configuration 4: Hexagonal-like lattice points
    # A hexagonal arrangement can have many 3-point lines
    hex_configs = [
        # 12 points in a hexagonal pattern
        [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], 
         [0,3], [1,3], [2,3]],
        # More symmetric
        [[0,0], [1,0], [-1,0], [0,1], [1,1], [-1,1], [0,-1], [1,-1], [-1,-1],
         [0,2], [0,-2], [2,0]],
    ]
    
    for pts in hex_configs:
        if not _validate(pts):
            continue
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = pts[:]
    
    # Local search: try to improve the best configuration found
    if best_points is not None:
        deadline2 = time.time() + 40
        current_best = best_score
        current_pts = best_points[:]
        
        for _ in range(2000):
            if time.time() > deadline2:
                break
            # Try moving one point
            i = random.randint(0, 11)
            old_p = current_pts[i][:]
            # Try small perturbations
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_p = [old_p[0] + dx, old_p[1] + dy]
                    if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
                        continue
                    test_pts = current_pts[:]
                    test_pts[i] = new_p
                    if not _validate(test_pts):
                        continue
                    score = _count_ordinary_lines(test_pts)
                    if score < current_best:
                        current_best = score
                        current_pts = test_pts[:]
                        break
                else:
                    continue
                break
    
    if best_points is None:
        # Fallback: original configuration
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_score = _count_ordinary_lines(best_points)
    
    return best_points

def _try_random_search():
    """Random search for good configurations."""
    deadline = time.time() + 60
    best_score = 10**9
    best_points = None
    
    # Start from some seed configurations
    seeds = [
        # 11 on a line, 1 off
        [[i, 0] for i in range(11)] + [[0, 1]],
        # 4x3 grid
        [[i, j] for i in range(4) for j in range(3)],
        # Parabola
        [[i, i*i] for i in range(11)] + [[0, 1]],
    ]
    
    current_best_pts = None
    for seed in seeds:
        if _validate(seed):
            score = _count_ordinary_lines(seed)
            if score < best_score:
                best_score = score
                best_points = seed[:]
                current_best_pts = seed[:]
    
    if current_best_pts is None:
        current_best_pts = seeds[0]
    
    # Local search from best seed
    for _ in range(5000):
        if time.time() > deadline:
            break
        i = random.randint(0, 11)
        old_p = current_best_pts[i][:]
        # Random perturbation
        scale = random.choice([1, 2, 3, 5, 10])
        dx = random.randint(-scale, scale)
        dy = random.randint(-scale, scale)
        if dx == 0 and dy == 0:
            continue
        new_p = [old_p[0] + dx, old_p[1] + dy]
        if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
            continue
        test_pts = current_best_pts[:]
        test_pts[i] = new_p
        if not _validate(test_pts):
            continue
        score = _count_ordinary_lines(test_pts)
        if score < best_score:
            best_score = score
            best_points = test_pts[:]
            current_best_pts = test_pts[:]
    
    if best_points is None:
        best_points = seeds[0]
    
    return best_points

def construct():
    """
    Construct 12 integer points with a minimized number of ordinary lines.
    Uses parabola-based configuration and local search.
    """
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Try structured approach first
    best_pts = _try_parabola_config()
    
    if time.time() > deadline:
        return best_pts
    
    # Try random search as well
    random_pts = _try_random_search()
    
    # Pick the better one
    score_structured = _count_ordinary_lines(best_pts)
    score_random = _count_ordinary_lines(random_pts)
    
    if score_random < score_structured:
        best_pts = random_pts
    
    return best_pts
# EVOLVE-BLOCK-END
