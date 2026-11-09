import time
import sys
import math
from math import gcd
from collections import defaultdict
import random

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line (as a normalized tuple) to the set of point indices on it
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            # Normalize (a, b, c) so that gcd(a,b,c) = 1 and a > 0 or (a == 0 and b > 0)
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return None
    # Check bounds
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    # Check not all collinear
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _get_lines_for_points(points):
    """Return dict mapping line key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    return line_map

def _count_ordinary_from_lines(line_map):
    """Count ordinary lines from a precomputed line_map."""
    return sum(1 for line, pts in line_map.items() if len(pts) == 2)

def _snap_candidates(points, i, line_map):
    """
    Generate candidate positions for point i by snapping to existing lines.
    For each line with >= 2 other points, find integer points on that line
    that are not already occupied.
    """
    n = len(points)
    candidates = set()
    for line_key, pts_on_line in line_map.items():
        if i in pts_on_line:
            pts_on_line = pts_on_line - {i}
        else:
            pass
        if len(pts_on_line) < 2:
            continue
        # Get the line equation a*x + b*y = c
        a, b, c = line_key
        # Find two points on this line to get direction
        pts_list = list(pts_on_line)
        if len(pts_list) < 2:
            continue
        p1 = points[pts_list[0]]
        p2 = points[pts_list[1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        g = gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
        # Direction vector is (dx, dy), but we need to be careful with sign
        # Actually, let's use the line equation directly
        # a*x + b*y = c, find integer solutions
        # We need gcd(a,b) | c for solutions to exist (should be true since points are on it)
        ga, gb = a, b
        gg = gcd(abs(ga), abs(gb))
        if gg == 0:
            continue
        ga_n = ga // gg
        gb_n = gb // gg
        c_n = c // gg
        if c_n * gg != c:
            continue  # No integer solution
        # Find one particular solution
        # ga_n * x + gb_n * y = c_n
        # Extended Euclidean
        def egcd(a, b):
            if b == 0:
                return (1, 0, a)
            x1, y1, g = egcd(b, a % b)
            return (y1, x1 - (a // b) * y1, g)
        
        x0, y0, g_val = egcd(ga_n, gb_n)
        if g_val != 1:
            continue
        # Particular solution: (x0 * c_n, y0 * c_n)
        px = x0 * c_n
        py = y0 * c_n
        # General solution: x = px + gb_n * t, y = py - ga_n * t
        # Try t values to get points near existing points
        existing = set(tuple(p) for j, p in enumerate(points) if j != i)
        # Try t values around 0 and around existing points
        t_range = range(-50, 51)
        for t in t_range:
            nx = px + gb_n * t
            ny = py - ga_n * t
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                if (nx, ny) not in existing:
                    candidates.add((nx, ny))
    return list(candidates)

def _generate_parabola_config(a, b, c, d, xs):
    """Generate 12 points on parabola y = a*x^2 + b*x + c + d*offset for 11 points, plus 12th point."""
    points = []
    for x in xs:
        y = a * x * x + b * x + c
        points.append([x, y])
    # 12th point: try to place it to maximize collinearities
    # For now, place at a position that might create 3-point lines
    if len(points) == 11:
        # Try several positions for the 12th point
        best_pt = [0, 0]
        best_cnt = float('inf')
        for tx in range(-5, 6):
            for ty in range(-5, 6):
                trial = points + [[tx, ty]]
                if len(set(tuple(p) for p in trial)) != 12:
                    continue
                cnt = _count_ordinary_lines(trial)
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_pt = [tx, ty]
        points.append(best_pt)
    return points

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Uses candidate configurations including parabola-based, local search,
    and simulated annealing with line-snapping moves.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s limit)
    
    # Candidate configurations
    candidates = []
    
    # Config 1: 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 11 on line + 1 off
    pts3 = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts3)
    
    # Config 4: 6 on x-axis, 6 on y-axis (cross)
    pts4 = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)]
    candidates.append(pts4)
    
    # Config 5: Points on two parallel lines (6 each)
    pts5 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
    candidates.append(pts5)
    
    # Config 8: 4 points on each of 3 concurrent lines (through origin)
    pts8 = [[0, 0]]
    for i in range(1, 5):
        pts8.append([i, 0])
        pts8.append([i, i])
        pts8.append([i, 2*i])
    candidates.append(pts8[:12])
    
    # Config 9: 2 lines through origin, 5 points each + 1 more
    pts9 = [[0, 0]]
    for i in range(1, 6):
        pts9.append([i, 0])
        pts9.append([i, i])
    pts9.append([1, 2])
    candidates.append(pts9)
    
    # Config 10: 10 on line + 2 off
    for off1_x, off1_y, off2_x, off2_y in [(0,1,1,1), (0,1,2,1), (0,1,0,2), (1,1,2,1), (0,1,1,2)]:
        pts10 = [[i, 0] for i in range(10)] + [[off1_x, off1_y], [off2_x, off2_y]]
        candidates.append(pts10)
    
    # Config 11: 9 on line + 3 off
    pts11 = [[i, 0] for i in range(9)] + [[0, 1], [1, 1], [2, 1]]
    candidates.append(pts11)
    
    # Config 12: 8 on line + 4 off
    pts12 = [[i, 0] for i in range(8)] + [[0, 1], [1, 1], [2, 1], [3, 1]]
    candidates.append(pts12)
    
    # Config 13: Two parallel lines with 6 each, shifted
    pts13 = [[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)]
    candidates.append(pts13)
    
    # Grid variants with different spacing
    for sx, sy in [(1,1), (1,2), (2,1), (2,2), (3,1), (1,3)]:
        pts_grid = [[x*sx, y*sy] for x in range(4) for y in range(3)]
        candidates.append(pts_grid)
    
    # Points on a parabola y=x^2
    pts_para = [[x, x*x] for x in range(12)]
    candidates.append(pts_para)
    
    # Parallel lines with different shifts
    for shift in range(0, 4):
        pts_par = [[i, 0] for i in range(6)] + [[i+shift, 1] for i in range(6)]
        candidates.append(pts_par)
    
    # Concurrent lines through origin
    pts_conc = [[0, 0]]
    for i in range(1, 4):
        pts_conc.append([i, 0])
        pts_conc.append([i, i])
        pts_conc.append([i, 2*i])
    pts_conc.append([1, 3])
    pts_conc.append([2, 6])
    candidates.append(pts_conc)
    
    # 5 lines through origin
    pts_5line = [[0, 0]]
    slopes = [(0,1), (1,1), (2,1), (1,2), (3,1)]
    for dy, dx in slopes:
        for t in range(1, 3):
            pts_5line.append([t*dx, t*dy])
    pts_5line.append([3, 3])
    candidates.append(pts_5line[:12])
    
    # NEW: Parabola-based configurations with different x-sets
    # Use symmetric x-values to create more collinearities
    x_sets = [
        list(range(-5, 7)),       # -5 to 6
        list(range(-6, 6)),       # -6 to 5
        list(range(-4, 8)),       # -4 to 7
        list(range(0, 12)),       # 0 to 11
        list(range(-3, 9)),       # -3 to 8
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    ]
    
    for xs in x_sets:
        if len(xs) != 12:
            continue
        pts_p = [[x, x*x] for x in xs]
        if len(set(tuple(p) for p in pts_p)) == 12:
            candidates.append(pts_p)
    
    # Parabola with different coefficients
    for a in [1, 2, -1]:
        for b in [-2, -1, 0, 1, 2]:
            for c in [-2, -1, 0, 1, 2]:
                pts_p = [[x, a*x*x + b*x + c] for x in range(12)]
                if len(set(tuple(p) for p in pts_p)) == 12:
                    candidates.append(pts_p)
    
    # Find the best configuration from candidates
    best_points = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        count = _try_config(pts)
        if count is not None and count < best_count:
            best_count = count
            best_points = [list(p) for p in pts]
    
    # If no valid config found, fallback
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_count = 11
    
    # Phase 1: Local search with small perturbations
    if best_points is not None and time.time() < deadline - 20:
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for i in range(12):
                if time.time() > deadline - 20:
                    break
                orig_x, orig_y = best_points[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
    
    # Phase 2: Simulated annealing with line-snapping
    if best_points is not None and time.time() < deadline - 5:
        current_points = [list(p) for p in best_points]
        current_count = best_count
        line_map = _get_lines_for_points(current_points)
        
        T = 10.0  # Initial temperature
        T_min = 0.1
        cooling = 0.995
        max_iter = 5000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Choose move type: snap (60%) or random (40%)
            if random.random() < 0.6:
                # Line snapping: pick a point and snap it to an existing line
                i = random.randint(0, 11)
                snap_cands = _snap_candidates(current_points, i, line_map)
                if snap_cands:
                    new_pt = list(random.choice(snap_cands))
                else:
                    # Fallback to random move
                    orig_x, orig_y = current_points[i]
                    new_pt = [orig_x + random.randint(-8, 8), orig_y + random.randint(-8, 8)]
            else:
                # Random move
                i = random.randint(0, 11)
                orig_x, orig_y = current_points[i]
                new_pt = [orig_x + random.randint(-8, 8), orig_y + random.randint(-8, 8)]
            
            if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                continue
            if any(tuple(new_pt) == tuple(current_points[j]) for j in range(12) if j != i):
                continue
            
            trial = [list(p) for p in current_points]
            trial[i] = new_pt
            count = _try_config(trial)
            if count is None:
                continue
            
            delta = count - current_count
            if delta < 0:
                # Accept improvement
                current_points = trial
                current_count = count
                line_map = _get_lines_for_points(current_points)
                if current_count < best_count:
                    best_points = [list(p) for p in current_points]
                    best_count = current_count
            else:
                # Accept worse with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    current_points = trial
                    current_count = count
                    line_map = _get_lines_for_points(current_points)
            
            T = max(T_min, T * cooling)
    
    # Phase 3: Final local polish
    if best_points is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                orig_x, orig_y = best_points[i]
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        if any(tuple(new_pt) == tuple(best_points[j]) for j in range(12) if j != i):
                            continue
                        trial = [list(p) for p in best_points]
                        trial[i] = new_pt
                        count = _try_config(trial)
                        if count is not None and count < best_count:
                            best_points = trial
                            best_count = count
                            improved = True
    
    return best_points
