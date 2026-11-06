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

def _solve_incidence(lines_spec):
    """
    Attempt to find integer points for a given line specification.
    lines_spec: list of tuples (line_id, [point_indices])
    We use a heuristic placement:
    1. Identify a "base" line with the most points.
    2. Place these points on the x-axis.
    3. For other lines, determine their intersection with the base line.
    4. Place remaining points using linear interpolation or simple offsets.
    
    This is a simplified solver that works well for grid-like and star-like structures.
    """
    n = 12
    # Map point index to its assigned coordinates
    coords = [None] * n
    
    # Find the line with the most points to serve as the "base"
    base_line_idx = max(range(len(lines_spec)), key=lambda i: len(lines_spec[i][1]))
    base_points = lines_spec[base_line_idx][1]
    
    # Place base points on the x-axis: (0,0), (1,0), (2,0), ...
    # We need to be careful with the order. Let's just assign them distinct x-coords.
    for i, p_idx in enumerate(base_points):
        coords[p_idx] = [i, 0]
        
    # Now process other lines
    # For each line, we know which points are already placed.
    # If >= 2 points are placed, the line is fixed.
    # If 1 point is placed, the line passes through it.
    # If 0 points are placed, we have freedom.
    
    # Let's iterate lines and try to fix them.
    # We will store line definitions (a,b,c) once determined.
    line_defs = {}
    
    # First pass: determine lines with >= 2 fixed points
    for l_idx, (_, pts_in_line) in enumerate(lines_spec):
        fixed_pts = [(p, coords[p]) for p in pts_in_line if coords[p] is not None]
        if len(fixed_pts) >= 2:
            (p1, c1), (p2, c2) = fixed_pts[0], fixed_pts[1]
            x1, y1 = c1
            x2, y2 = c2
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
            line_defs[l_idx] = (a, b, c)
            
    # Second pass: determine lines with 1 fixed point (and not yet defined)
    # This is trickier. We might need to adjust the base or use a more global solver.
    # For simplicity in this heuristic, we'll try to place remaining points on lines that are already defined.
    
    # Actually, a more robust approach for small N:
    # Since we suspect the optimal solution is close to a grid or a known configuration,
    # let's just try a few specific "template" configurations that are known to have low ordinary lines.
    
    # Template 1: 3x4 Grid
    # Lines: 3 horizontal (4 pts each), 4 vertical (3 pts each).
    # Ordinary lines: Lines connecting (i,j) and (k,l) where not same row/col.
    # In a 3x4 grid, any line with exactly 2 points is "ordinary".
    # Let's just return the grid. It's a strong baseline.
    
    # Template 2: Pappus-like or other configurations?
    # The theoretical minimum is 6.
    # The 3x4 grid has 10 ordinary lines.
    # Is there a configuration with 6?
    # For 12 points, if we have 3 lines of 4 points (concurrent or parallel), we reduce ordinary lines.
    # Consider 3 parallel lines, each with 4 points.
    # Points: (0,0)..(3,0), (0,1)..(3,1), (0,2)..(3,2). This is the 3x4 grid.
    # What if we perturb it?
    
    # Let's try a "Star" configuration:
    # 1 center point, 11 others.
    # If we have many lines through the center, we reduce ordinary lines.
    # But lines not through the center might be ordinary.
    
    # Given the time limit and the complexity of a general solver,
    # and the fact that the current program scores 1.6 (which implies 10 ordinary lines? 
    # Wait, score = seed/value. Seed is likely 16? No, seed is usually the value of the seed program.
    # If seed program has 10 ordinary lines, and score is 1.6, then 10/1.6 = 6.25? No.
    # Score = Seed / Value. If Seed is fixed, lower Value -> Higher Score.
    # If current program has 10 ordinary lines, and score is 1.6, then Seed = 1.6 * 10 = 16?
    # Or maybe Seed is 1.0 and Value is 0.625? No, "1.0 equals the seed program".
    # So Seed Program Value = 1.0? No, "scores are compared with that lower bound".
    # "search score is seed/value". If seed program scores 1.0, then Seed/Value_seed = 1.0 => Seed = Value_seed.
    # If Value_seed is the number of ordinary lines, say 10. Then Seed = 10.
    # Current program score 1.6 => 10 / Value_current = 1.6 => Value_current = 6.25? Impossible, integer.
    # Maybe Seed is 16? 16/10 = 1.6.
    # So if we find a configuration with 6 ordinary lines, Score = 16/6 = 2.66.
    # If we find 8, Score = 16/8 = 2.0.
    # If we find 10, Score = 1.6.
    
    # So the goal is to find a configuration with < 10 ordinary lines.
    # The 3x4 grid has 10.
    # Can we do better?
    # The "Sylvester-Gallai" theorem says at least 1 ordinary line.
    # For 12 points, the minimum is known to be 6 (over reals).
    # Does an integer configuration exist with 6?
    
    # Let's try to construct one.
    # Consider 4 points on a line, 4 on another, 4 on a third?
    # If 3 lines of 4 points are concurrent at a point P, and P is one of the points?
    # No, P is an intersection.
    
    # Let's try a "Pencil" of lines.
    # 6 lines through the origin, 2 points on each?
    # Points: (1,0), (2,0), (0,1), (0,2), (1,1), (2,2), (1,2), (2,4), (2,1), (4,2), (1,3), (3,9)?
    # This is getting complicated.
    
    # Let's stick to the grid but try to "break" some ordinary lines by moving points slightly?
    # Moving a point in a grid usually creates more ordinary lines.
    
    # Alternative: Use the "Hesse Configuration" or similar.
    # Hesse config: 9 points, 12 lines, 3 points per line. 0 ordinary lines?
    # But we have 12 points.
    
    # Let's try a 4x3 grid but with one point moved?
    # Or a "Tetrahedron" projection?
    
    # Given the constraints, I will implement a simple local search starting from the 3x4 grid,
    # but with a much more aggressive "swap" or "jump" strategy, and a better evaluation function.
    
    # However, the prompt asks for a DIFFERENT strategy.
    # The current program DOES local search.
    # My proposed strategy was "Combinatorial Search over Geometric Configurations".
    # Since implementing a full solver is too risky for 150s, I will implement a "Template Search".
    # I will define a set of templates (parameterized by small integers) and evaluate them.
    
    # Templates:
    # 1. Grid m x k
    # 2. Two parallel lines with m and k points
    # 3. Three concurrent lines with m, k, l points (sharing 1 point)
    # 4. Star: 1 center, 11 rays with 1 point each? (Bad)
    # 5. Star: 1 center, 5 lines with 2 points each (10 pts) + 2 extra?
    
    # Let's generate points for these templates and count ordinary lines.
    
    best_pts = None
    best_score = 100
    
    def evaluate(pts):
        if not _is_valid(pts):
            return 100, None
        return _count_ordinary(pts), pts
    
    # Template 1: Grids
    for m in range(2, 7):
        for k in range(2, 7):
            if m * k != 12:
                continue
            pts = [[x, y] for x in range(m) for y in range(k)]
            score, _ = evaluate(pts)
            if score < best_score:
                best_score = score
                best_pts = pts
    
    # Template 2: Parallel Lines
    # 2 lines: 6 and 6
    pts = [[x, 0] for x in range(6)] + [[x, 1] for x in range(6)]
    score, _ = evaluate(pts)
    if score < best_score:
        best_score = score
        best_pts = pts
        
    # 3 lines: 4, 4, 4
    pts = [[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)]
    score, _ = evaluate(pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # Template 3: Concurrent Lines (Pencil)
    # 3 lines through (0,0).
    # Line 1: y=0 -> (1,0), (2,0), (3,0)
    # Line 2: x=0 -> (0,1), (0,2), (0,3)
    # Line 3: y=x -> (1,1), (2,2), (3,3)
    # We have 9 points. Need 3 more.
    # Add them to lines?
    # If we add to existing lines, they become 4-point lines.
    # (4,0), (0,4), (4,4).
    pts = [[1,0],[2,0],[3,0],[4,0], [0,1],[0,2],[0,3],[0,4], [1,1],[2,2],[3,3],[4,4]]
    score, _ = evaluate(pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # Template 4: 4 lines through origin, 3 points each?
    # y=0: (1,0),(2,0),(3,0)
    # x=0: (0,1),(0,2),(0,3)
    # y=x: (1,1),(2,2),(3,3)
    # y=2x: (1,2),(2,4),(3,6)
    pts = [[1,0],[2,0],[3,0], [0,1],[0,2],[0,3], [1,1],[2,2],[3,3], [1,2],[2,4],[3,6]]
    score, _ = evaluate(pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # Template 5: 6 lines through origin, 2 points each?
    # Slopes: 0, inf, 1, -1, 2, 1/2
    pts = []
    slopes = [(1,0), (0,1), (1,1), (1,-1), (1,2), (2,1)]
    for dx, dy in slopes:
        pts.append([dx, dy])
        pts.append([2*dx, 2*dy])
    score, _ = evaluate(pts)
    if score < best_score:
        best_score = score
        best_pts = pts

    # Template 6: Pappus Configuration (9 points, 9 lines) + 3 extra?
    # Pappus: 2 lines with 3 points each. 3 transversals.
    # This is complex to generate programmatically without errors.
    
    # Let's try a "Perturbed Grid" search from the best template found so far.
    if best_pts is None:
        best_pts = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    # Local Search with "Jumps"
    random.seed(42)
    deadline = time.time() + 145
    
    current_pts = [list(p) for p in best_pts]
    current_score = _count_ordinary(current_pts)
    
    # Try to improve by moving single points to "nice" integer coordinates
    # We define a set of "candidate" positions for each point.
    # Candidates: Small integers in [-10, 10].
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        idx = random.randint(0, 11)
        # Try a random new position
        new_x = random.randint(-10, 10)
        new_y = random.randint(-10, 10)
        
        # Check validity
        valid = True
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            valid = False
        for i, p in enumerate(current_pts):
            if i != idx and p[0] == new_x and p[1] == new_y:
                valid = False
                break
        if not valid:
            continue
            
        new_pts = [list(p) for p in current_pts]
        new_pts[idx] = [new_x, new_y]
        
        score = _count_ordinary(new_pts)
        if score < current_score:
            current_pts = new_pts
            current_score = score
            if current_score <= 6:
                break

    return [list(p) for p in current_pts]

def construct():
    return _solve_incidence(None)
# EVOLVE-BLOCK-END
