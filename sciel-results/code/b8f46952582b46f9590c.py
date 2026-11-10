"""Construct 12 integer points minimizing ordinary lines (lines with exactly 2 points)."""

from math import gcd
from itertools import combinations

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = -dy
            b = dx
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for count in line_counts.values():
        if count == 1:
            ordinary += 1
    return ordinary

def construct():
    # Hexagon + dual hexagram approach
    # Outer hexagon: 6 points, inner/rotated hexagram: 6 points
    
    # Configuration 1: Two concentric hexagons, one rotated 30 degrees
    # Using integer-friendly coordinates
    
    # Hexagon A (radius ~2): (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)
    # Hexagon B (radius ~1, rotated 30°): (0,1), (1,0), (0,-1), (-1,0) - only 4, need 6
    # Let me use a different scaling
    
    # Try: 6 points on y=±x lines and 6 on axes, forming a star pattern
    
    # Configuration: vertices of a regular 12-gon projected to integers
    # Or: use the (12_3) configuration from design theory
    
    # Let me try several candidates:
    
    # Candidate 1: 4x3 grid
    cand1 = [[x, y] for x in range(4) for y in range(3)]
    
    # Candidate 2: 3 concurrent lines with 4 points each
    cand2 = [
        [1, 0], [2, 0], [3, 0], [4, 0],
        [1, 1], [2, 2], [3, 3], [4, 4],
        [1, -1], [2, -2], [3, -3], [4, -4]
    ]
    
    # Candidate 3: Hexagon-like with symmetry
    # 6 on outer ring, 6 on inner ring, rotated
    # Outer: (3,0), (2,3), (-2,3), (-3,0), (-2,-3), (2,-3)
    # Inner: (0,2), (2,0), (0,-2), (-2,0) - only 4, add (1,1), (1,-1)?
    # Let me try: outer hexagon + center + 5 more
    
    # Candidate 4: Based on Pappus-like configuration
    # 3 lines, 4 points each, arranged so many cross-lines have 3+ points
    cand4 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    # This is 4x3 grid again
    
    # Candidate 5: Two parallel lines with 6 points each
    cand5 = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1]
    ]
    
    # Candidate 6: Hexagram - two overlapping triangles
    # Triangle 1: (0, 4), (-4, -2), (4, -2)
    # Triangle 2: (0, -4), (-4, 2), (4, 2)
    # Plus 6 more points at intersections or midpoints
    cand6 = [
        [0, 4], [-4, -2], [4, -2],
        [0, -4], [-4, 2], [4, 2],
        [0, 0], [-2, 0], [2, 0],
        [0, 2], [0, -2], [-2, 2]  # need to pick 6 more carefully
    ]
    # That's 12 but let me verify distinctness and try better
    
    # Candidate 7: Regular hexagon vertices + 6 more
    # Hex: (2,0), (1,2), (-1,2), (-2,0), (-1,-2), (1,-2)
    # Add: (0,1), (1,0), (0,-1), (-1,0), (0,3), (3,0)? 
    cand7 = [
        [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2], [1, -2],
        [0, 1], [1, 0], [0, -1], [-1, 0], [0, 3], [3, 0]
    ]
    
    # Candidate 8: Symmetric 12-point configuration
    # 4 on x-axis, 4 on y-axis, 4 on diagonals
    cand8 = [
        [1, 0], [2, 0], [3, 0], [4, 0],
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 1], [2, 2], [3, 3], [4, 4]
    ]
    
    # Candidate 9: 3x4 grid variant
    cand9 = [[x, y] for x in range(3) for y in range(4)]
    
    # Candidate 10: Star of David configuration
    # Two equilateral triangles overlapping
    # Triangle 1: (0, 3), (-3, -1), (3, -1)  
    # Triangle 2: (0, -3), (-3, 1), (3, 1)
    # Midpoints/intersections: (0,0), (1,1), (-1,1), (1,-1), (-1,-1), (0,2)?
    cand10 = [
        [0, 3], [-3, -1], [3, -1],
        [0, -3], [-3, 1], [3, 1],
        [0, 0], [1, 1], [-1, 1],
        [1, -1], [-1, -1], [0, 2]
    ]
    
    # Evaluate all candidates
    candidates = [cand1, cand2, cand5, cand6, cand7, cand8, cand9, cand10]
    
    best_points = None
    best_count = float('inf')
    
    for cand in candidates:
        # Check distinctness
        if len(set(tuple(p) for p in cand)) != 12:
            continue
        # Check bounds
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in cand):
            continue
        # Check not all collinear
        if len(cand) >= 3:
            (x1, y1), (x2, y2), (x3, y3) = cand[0], cand[1], cand[2]
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                # Check if all are collinear
                all_col = all((x2-x1)*(p[1]-y1) == (y2-y1)*(p[0]-x1) for p in cand[3:])
                if all_col:
                    continue
        count = _count_ordinary(cand)
        if count < best_count:
            best_count = count
            best_points = cand
    
    # Try local search from best candidate
    if best_points is not None:
        # Try small perturbations
        import random
        random.seed(42)
        current = [list(p) for p in best_points]
        current_count = best_count
        
        for _ in range(2000):
            idx = random.randint(0, 11)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_pts = [list(p) for p in current]
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            
            # Check validity
            if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in new_pts):
                continue
            if len(set(tuple(p) for p in new_pts)) != 12:
                continue
            
            new_count = _count_ordinary(new_pts)
            if new_count < current_count:
                current = new_pts
                current_count = new_count
    
    return current if best_points is not None else [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
