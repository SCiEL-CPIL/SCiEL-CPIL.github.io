import time
import sys
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Map each line to the set of points on it
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): normalize (a,b,c) for ax+by+c=0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            # Normalize: gcd
            from math import gcd
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    # Check distinct
    seen = set()
    for p in points:
        if p in seen:
            return None
        seen.add(p)
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return None
    # Check not all collinear
    if len(set((points[i][1] - points[0][1]) * (points[j][0] - points[0][0]) - 
                (points[i][0] - points[0][0]) * (points[j][1] - points[0][1])
                for i in range(12) for j in range(i+1, 12))) <= 1:
        return None
    return _count_ordinary_lines(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_count = 999
    best_points = None
    
    # Try various structured configurations
    
    # Config 1: 11 on line, 1 off (baseline: 11 ordinary lines)
    pts = [[i, 0] for i in range(11)] + [[0, 1]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 2: 6 on x-axis, 6 on y-axis (minus origin counted once)
    # (0,0),(1,0),(2,0),(3,0),(4,0),(5,0) and (0,1),(0,2),(0,3),(0,4),(0,5),(0,6)
    pts = [[i, 0] for i in range(6)] + [[0, j] for j in range(1, 7)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 3: 4x3 grid
    pts = [[i, j] for i in range(4) for j in range(3)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 4: 3x4 grid
    pts = [[i, j] for i in range(3) for j in range(4)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 5: Hexagonal-like with extra points
    # Center + 6 around + 6 more
    pts = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[2,0],[-2,0],[0,2],[0,-2],[2,1]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 6: Two lines of 6, offset
    pts = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 7: Three lines of 4
    pts = [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 8: Complete quadrilateral style - 4 lines with 4 points each
    # Lines: y=0, y=1, x=0, x=1 give 8 points, need 4 more
    pts = [[i, j] for i in range(2) for j in range(4)] + [[i, j] for i in range(2,4) for j in range(2)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 9: Try 5 on one line, 7 arranged to maximize collinearities
    # 5 on x-axis: (0,0)-(4,0)
    # 7 points: try to put them on 2-3 other lines with >=3 points
    # Line y=1: (0,1),(1,1),(2,1),(3,1) - 4 points
    # Line x=0: already have (0,0),(0,1), add (0,2),(0,3) - 4 points total on x=0
    # Remaining: 12-5-4-2=1 point... let me recount
    # (0,0),(1,0),(2,0),(3,0),(4,0) = 5
    # (0,1),(1,1),(2,1),(3,1) = 4, total 9
    # (0,2),(0,3) = 2, total 11
    # Need 1 more: (1,2)
    pts = [[0,0],[1,0],[2,0],[3,0],[4,0],[0,1],[1,1],[2,1],[3,1],[0,2],[0,3],[1,2]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 10: Use a finite field-like structure
    # GF(3) affine plane has 9 points, 12 lines of 3 points
    # Add 3 more points strategically
    pts = [[i, j] for i in range(3) for j in range(3)]
    # Add 3 points to create more 3-point lines
    # (3,0),(3,1),(3,2) - vertical line x=3 with 3 points
    pts += [[3,0],[3,1],[3,2]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 11: 4x3 grid with one point moved
    pts = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,3]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 12: Diagonal-heavy configuration
    # Points on y=x, y=x+1, y=x-1, y=0
    pts = [[0,0],[1,1],[2,2],[3,3],  # y=x, 4 pts
           [0,1],[1,2],[2,3],          # y=x+1, 3 pts
           [1,0],[2,1],[3,2],          # y=x-1, 3 pts
           [4,0],[5,0]]                # y=0, 2 pts (but (0,0),(1,0) already? No)
    # Wait, (0,0) is on both y=x and y=0. Let me recount: 4+3+3+2=12 but (0,0) counted twice
    pts = [[0,0],[1,1],[2,2],[3,3],[0,1],[1,2],[2,3],[1,0],[2,1],[3,2],[4,0],[5,0]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 13: Star configuration - multiple lines through common points
    # Center point + 5 lines through it with 2 other points each
    pts = [[0,0],  # center
           [1,0],[-1,0],  # horizontal
           [0,1],[0,-1],  # vertical
           [1,1],[-1,-1], # diagonal /
           [1,-1],[-1,1], # diagonal \
           [2,0],[0,2]]   # extra
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 14: Try to get close to 6 ordinary lines
    # Known: for n=12, min ordinary lines = 6
    # One construction: 6 points on a conic, but we need integers
    # Alternative: use the "Böröczky" type
    # Try: 4 points on each of 3 concurrent lines + 4 more
    pts = [[0,0],[1,0],[2,0],[3,0],  # line y=0
           [0,1],[0,2],[0,3],         # line x=0 (shares (0,0))
           [1,1],[2,2],[3,3],         # line y=x (shares (0,0))
           [4,0],[0,4],[4,4]]         # extra
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 15: 6 points on two parallel lines of 3
    pts = [[i, 0] for i in range(3)] + [[i, 1] for i in range(3)] + [[i, 2] for i in range(3)] + [[i, 3] for i in range(3)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 16: Try specific known good configurations
    # 3x4 with modified points
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3+dx,2+dy]]
            c = _try_config(pts)
            if c is not None and c < best_count:
                best_count = c
                best_points = [list(p) for p in pts]
    
    # Config 17: More systematic search around 4x3 grid
    base = [[i, j] for i in range(4) for j in range(3)]
    for idx in range(12):
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                pts = [list(p) for p in base]
                pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
                c = _try_config(pts)
                if c is not None and c < best_count:
                    best_count = c
                    best_points = [list(p) for p in pts]
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        if best_points is not None:
            return best_points
        return [[i, 0] for i in range(11)] + [[0, 1]]
    
    # Config 18: Try 2x6 grid
    pts = [[i, j] for i in range(6) for j in range(2)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 19: 6 points on one line, 6 on another non-parallel line
    pts = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 20: Try configurations with more 3-point lines
    # 4 lines of 3 points each, sharing points
    pts = [[0,0],[1,0],[2,0],  # line 1: y=0
           [0,1],[1,1],[2,1],  # line 2: y=1  
           [0,2],[1,2],[2,2],  # line 3: y=2
           [3,0],[3,1],[3,2]]  # line 4: x=3
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 21: More creative - use slopes to create many 3-point lines
    # Points: (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,3),(4,4),(5,5)
    pts = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,3],[4,4],[5,5]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 22: Try to maximize number of 3+ point lines
    # Use a 3x3 grid plus 3 points on a line through 3 grid points extended
    pts = [[i, j] for i in range(3) for j in range(3)] + [[3, 0], [3, 1], [3, 2]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 23: 3x3 grid + 3 points forming a triangle with grid points
    pts = [[i, j] for i in range(3) for j in range(3)] + [[3, 0], [0, 3], [3, 3]]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 24: Try 6+6 split with shared structure
    pts = [[i, 0] for i in range(6)] + [[i, i] for i in range(1, 7)]
    c = _try_config(pts)
    if c is not None and c < best_count:
        best_count = c
        best_points = [list(p) for p in pts]
    
    # Config 25: Systematic - try all 4x3 grid variants with one point perturbed
    base = [[i, j] for i in range(4) for j in range(3)]
    for idx in range(12):
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                pts = [list(p) for p in base]
                pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
                c = _try_config(pts)
                if c is not None and c < best_count:
                    best_count = c
                    best_points = [list(p) for p in pts]
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_points is not None:
        return best_points
    
    # Fallback
    return [[i, 0] for i in range(11)] + [[0, 1]]
