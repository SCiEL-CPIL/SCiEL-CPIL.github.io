"""
Minimize ordinary lines (lines containing exactly 2 of the 12 points)
for 12 integer points with |x|,|y| <= 1e9.

Strategy:
1. Use known combinatorial/geometric constructions that minimize ordinary lines.
2. For n=12, the theoretical minimum over reals is 6 (from the Orchard-Planting problem / Green-Tao related results).
3. We try to find a configuration of 12 rational/integer points achieving as few ordinary lines as possible.

Known approach: Use points on a conic or structured arrangements.
One classic construction uses points on a parabola or a grid with careful selection.

Actually, for small n, the minimum number of ordinary lines is known:
n=3: 3
n=4: 3 (if not all collinear; a parallelogram has 3 ordinary lines? Let's check: 4 points in convex position - each pair defines a line, 6 lines total, but in a parallelogram, 2 pairs are parallel but not collinear... wait, each line through 2 points: in a general quadrilateral, all 6 lines are ordinary. In a parallelogram, the diagonals intersect but don't create 3-point lines. So still 6? No wait, 4 points: C(4,2)=6 pairs. If no 3 are collinear, all 6 lines are ordinary. But we can have 3 collinear: then the line through those 3 is not ordinary, and the other lines... 

Let me think about known minima. The problem of minimum ordinary lines for n points:
- n=3: 3
- n=4: 3 (take 3 collinear + 1 off: the line with 3 is not ordinary; the 3 lines from the 4th point to each of the 3 are ordinary if the 4th isn't on those lines... wait, if 3 are collinear and 1 is off, then lines from the off-point to each of the 3: these 3 lines each contain exactly 2 points (the off-point and one of the 3), so 3 ordinary lines. The line through the 3 collinear points is not ordinary. So 3 ordinary lines.)
- n=5: 4? 
- Actually the known sequence for minimum ordinary lines: 3,3,4,6,6,8,... for n=3,4,5,6,7,8...

For n=12, I believe the minimum might be around 6-10.

Let me use a different approach: 
- Start with a good initial configuration
- Use local search (move points to reduce ordinary line count)
- Use simulated annealing or hill climbing

But the key insight: we need integer points. Let's use a structured approach.

One idea: use points on a small grid and then optimize.
Another: use the fact that if many points are collinear, we reduce ordinary lines.

Let me try a construction: 
- Put points on lines that intersect at common points.

Actually, let me use a computational approach:
1. Generate candidate configurations from known patterns
2. Use local search to improve

For the local search, I'll represent points as integers and try small perturbations.

Let me code a solution that:
1. Tries several initial configurations
2. Uses local search to minimize ordinary lines
3. Returns the best found

Given the 150s deadline, I'll use time-bounded search.
"""

import time
import random
import math
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Use a set of normalized line representations
    # A line through (x1,y1) and (x2,y2) can be represented by (A,B,C) where
    # Ax + By + C = 0, normalized so that gcd(A,B,C)=1 and A>0 or (A==0 and B>0)
    
    line_map = {}  # line_key -> count of points on this line
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Line through (x1,y1) and (x2,y2): 
            # (y2-y1)(x - x1) - (x2-x1)(y - y1) = 0
            # A = dy, B = -dx, C = dx*y1 - dy*x1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            # Normalize
            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue  # shouldn't happen for distinct points
            A //= g
            B //= g
            C //= g
            
            # Ensure canonical form: first non-zero of (A,B) is positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_map[key] = line_map.get(key, 0) + 1
    
    # Now count lines with exactly 2 points
    # But we need to be careful: each line is counted by the number of pairs on it
    # A line with k points has C(k,2) pairs. So we need to invert this.
    
    # Actually, line_map counts pairs, not points. Let me reconsider.
    # For a line with k points, it appears C(k,2) times in line_map.
    # We want to count lines where k=2, which appear exactly once.
    
    # But we can't directly recover k from the pair count without knowing all lines.
    # Better approach: for each unique line, count how many points lie on it.
    
    # Let's use a different method: group points by line
    # For each pair, get the line, then for each line, count distinct points.
    
    point_on_line = {}  # line_key -> set of point indices
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            g = math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in point_on_line:
                point_on_line[key] = set()
            point_on_line[key].add(i)
            point_on_line[key].add(j)
    
    ordinary_count = 0
    for key, pts in point_on_line.items():
        if len(pts) == 2:
            ordinary_count += 1
    
    return ordinary_count


def _are_all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        x2, y2 = points[i]
        if (x2 - x0) * (y1 - y0) != (x1 - x0) * (y2 - y0):
            return False
    return True


def _local_search(points, max_iters=5000, deadline=None):
    """Local search to minimize ordinary lines."""
    n = len(points)
    best = [list(p) for p in points]
    best_score = _count_ordinary_lines(best)
    
    if deadline is not None and time.time() > deadline:
        return best, best_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, n-1)
        old_x, old_y = best[idx]
        
        # Try a random new position in a small neighborhood
        # Use larger steps occasionally
        step = random.choice([1, 1, 2, 3, 5, 10, 20])
        new_x = old_x + random.randint(-step, step)
        new_y = old_y + random.randint(-step, step)
        
        # Ensure bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        
        # Check distinctness
        new_points = [list(p) for p in best]
        new_points[idx] = [new_x, new_y]
        
        # Check if any point is duplicated
        seen = set()
        valid = True
        for p in new_points:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        
        if not valid:
            continue
        
        # Check not all collinear
        if _are_all_collinear(new_points):
            continue
        
        score = _count_ordinary_lines(new_points)
        
        if score < best_score:
            best = new_points
            best_score = score
    
    return best, best_score


def _try_construction():
    """Try various constructions and return the best configuration."""
    candidates = []
    
    # Construction 1: 3x4 grid (baseline)
    grid = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid)
    
    # Construction 2: Points on a conic-like structure
    # Use points that create many 3-point lines
    
    # Construction 3: Based on known minimal configurations
    # Try a configuration with multiple collinear triples
    
    # Let's try: 4 points on one line, 4 on another, 4 on a third
    # Arranged so lines intersect nicely
    
    # For now, generate several random configurations and optimize
    
    random.seed(42)
    for trial in range(20):
        # Generate random points in a small range to start
        pts = []
        for _ in range(12):
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if [x, y] not in pts:
                pts.append([x, y])
        
        # Fill if needed
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in pts:
                pts.append([x, y])
        
        if len(pts) == 12 and not _are_all_collinear(pts):
            candidates.append(pts)
    
    # Try structured constructions
    # Construction: points forming a grid with some modifications
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid  
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Try a configuration based on the Hesse configuration or similar
    # For 12 points, try to use 4 lines of 3 points each, with intersections
    
    # Let's use a computational approach: start with good seeds and optimize
    
    best_points = None
    best_score = float('inf')
    
    deadline = time.time() + 120  # Leave margin for the 150s limit
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Scale up to use more of the coordinate space if needed
        # For now, keep small coordinates
        
        # Run local search
        optimized, score = _local_search(cand, max_iters=2000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_points = optimized
    
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary_lines(best_points)
    
    return best_points, best_score


def construct():
    """Construct 12 integer points minimizing ordinary lines."""
    start = time.time()
    
    points, score = _try_construction()
    
    # Ensure validity
    assert len(points) == 12
    assert len(set(tuple(p) for p in points)) == 12
    for p in points:
        assert abs(p[0]) <= 10**9
        assert abs(p[1]) <= 10**9
    assert not _are_all_collinear(points)
    
    return points
