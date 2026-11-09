import time
import sys
from math import gcd

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Map each line (as a normalized tuple) to the set of point indices on it
    from collections import defaultdict
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
            # Also normalize the line by using the "leftmost/bottommost" point as reference
            # A line can be represented by (dx, dy, x0*dx - y0*dy) where (x0,y0) is a point on the line
            # But to make it canonical, we need to handle sign and offset
            # Better: use the line equation ax + by = c where a,b are coprime and c is normalized
            # For direction (dx, dy), the normal is (dy, -dx), so line: dy*x - dx*y = c
            # c = dy*x1 - dx*y1
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

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Tests several known configurations and performs local search to improve.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Candidate configurations
    candidates = []
    
    # Config 1: 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 11 on line + 1 off (current baseline)
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
    
    # Config 10: 10 on line + 2 off, with the 2 off points chosen to create collinearities
    # Off points at (0,1) and (1,1): line through them is y=1
    # Line through (0,1) and (i,0): slope = -1/i
    # Line through (1,1) and (j,0): slope = -1/(j-1)
    # For i=1: line through (0,1) and (1,0) has slope -1
    # For j=2: line through (1,1) and (2,0) has slope -1 -> same line!
    # So (0,1), (1,0), (1,1), (2,0) are not all on one line, but (0,1),(1,0) and (1,1),(2,0) are on parallel lines
    # Actually let me just test various off-point positions
    for off1_x, off1_y, off2_x, off2_y in [(0,1,1,1), (0,1,2,1), (0,1,0,2), (1,1,2,1), (0,1,1,2)]:
        pts10 = [[i, 0] for i in range(10)] + [[off1_x, off1_y], [off2_x, off2_y]]
        candidates.append(pts10)
    
    # Config 11: 9 on line + 3 off, arranged to create collinearities
    # Try: 9 on x-axis, 3 off at (0,1),(1,1),(2,1) -> line y=1 has 3 points
    pts11 = [[i, 0] for i in range(9)] + [[0, 1], [1, 1], [2, 1]]
    candidates.append(pts11)
    
    # Config 12: 8 on line + 4 off
    pts12 = [[i, 0] for i in range(8)] + [[0, 1], [1, 1], [2, 1], [3, 1]]
    candidates.append(pts12)
    
    # Config 13: Two parallel lines with 6 each, but shifted to create diagonal collinearities
    # (i, 0) for i=0..5 and (i+1, 1) for i=0..5 -> (1,1),(2,1),...,(6,1)
    pts13 = [[i, 0] for i in range(6)] + [[i+1, 1] for i in range(6)]
    candidates.append(pts13)
    
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
    
    # Local search: try to improve by moving one point at a time
    if best_points is not None and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                orig_x, orig_y = best_points[i]
                # Try small perturbations
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = [orig_x + dx, orig_y + dy]
                        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                            continue
                        # Check distinctness
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
