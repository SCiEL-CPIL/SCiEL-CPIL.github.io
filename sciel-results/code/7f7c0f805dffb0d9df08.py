"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if points are valid: 12 distinct, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    for i in range(2, len(points)):
        x, y = points[i]
        if A * x + B * y + C != 0:
            return True
    return False

def construct():
    # Deterministic construction: try known configurations and return best
    # These are specific hand-crafted configurations that aim for 6 ordinary lines
    
    # Configuration A: 4x3 grid (classic)
    pts_A = [(i, j) for i in range(4) for j in range(3)]
    
    # Configuration B: 3x4 grid
    pts_B = [(i, j) for i in range(3) for j in range(4)]
    
    # Configuration C: Two parallel lines with 6 points each, aligned
    pts_C = [(i, 0) for i in range(6)] + [(i, 1) for i in range(6)]
    
    # Configuration D: Two parallel lines with 6 points each, offset
    pts_D = [(i, 0) for i in range(6)] + [(i+1, 1) for i in range(6)]
    
    # Configuration E: 6 points on x-axis, 6 on y=1 with different spacing
    pts_E = [(i, 0) for i in range(6)] + [(i*2, 1) for i in range(6)]
    
    # Configuration F: Points on parabola y = x^2
    pts_F = [(t, t*t) for t in range(-3, 5)]
    
    # Configuration G: 3x4 grid with shifted middle row
    pts_G = [(i, 0) for i in range(4)] + [(i+1, 1) for i in range(4)] + [(i, 2) for i in range(4)]
    
    # Configuration H: 4 points on each of 3 parallel lines
    pts_H = [(i, 0) for i in range(4)] + [(i, 1) for i in range(4)] + [(i, 2) for i in range(4)]
    
    # Configuration I: 12 points in a hexagonal-like arrangement
    pts_I = [(0,0),(1,0),(2,0),(3,0),
             (0,1),(1,1),(2,1),(3,1),
             (0,2),(1,2),(2,2),(3,2)]
    
    # Configuration J: 6 on x-axis, 6 on diagonal
    pts_J = [(i, 0) for i in range(6)] + [(i, i) for i in range(1, 7)]
    
    # Configuration K: 12 points forming two triangles (one up, one down)
    pts_K = [(0,0),(2,0),(1,1),
             (3,0),(5,0),(4,1),
             (0,3),(2,3),(1,2),
             (3,3),(5,3),(4,2)]
    
    # Configuration L: 12 points on a circle approximated by integers
    pts_L = [(6,0),(5,3),(2,5),(-2,5),(-5,3),(-6,0),
             (-5,-3),(-2,-5),(2,-5),(5,-3),(0,6),(0,-6)]
    # Scale down to fit
    pts_L = [(3,0),(3,2),(1,3),(-1,3),(-3,2),(-3,0),
             (-3,-2),(-1,-3),(1,-3),(3,-2),(0,3),(0,-3)]
    
    candidates = [pts_A, pts_B, pts_C, pts_D, pts_E, pts_F, pts_G, pts_H, pts_I, pts_J, pts_K, pts_L]
    
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                break
    
    if best_points is not None:
        return best_points
    
    # Fallback: 4x3 grid
    return [[x, y] for x in range(4) for y in range(3)]
# EVOLVE-BLOCK-END
