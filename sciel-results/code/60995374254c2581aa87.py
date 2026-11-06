"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use normalized line representation: (A, B, C) for Ax + By + C = 0
    # with gcd(A,B,C)=1 and A>0 or (A==0 and B>0) or (A==0 and B==0 and C>0)
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2):
            # A = y2-y1, B = x1-x2, C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            # Make canonical: first nonzero positive
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
    # Not all collinear
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
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_points = None
    best_count = float('inf')
    
    # Try several structural configurations
    
    # Configuration 1: 6 points on y=0, 6 points on y=1, with x-coords chosen for symmetry
    # Try: (0,0),(1,0),(2,0),(3,0),(4,0),(5,0) and (0,1),(1,1),(2,1),(3,1),(4,1),(5,1)
    # This is a 6x2 grid - likely many ordinary lines
    
    # Configuration 2: Use a more structured approach
    # Try points on two parabolas: y=x^2 and y=x^2+1 for x=0..5
    for shift in [1, 2, 3]:
        pts = [(x, x*x) for x in range(6)] + [(x, x*x+shift) for x in range(6)]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # Configuration 3: 4 points on x-axis, 4 on y-axis, 4 more
    # Try: (0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(3,3)
    for variant in range(5):
        if time.time() > deadline:
            break
        if variant == 0:
            pts = [(0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(3,3)]
        elif variant == 1:
            pts = [(0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(1,3)]
        elif variant == 2:
            pts = [(0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(3,1)]
        elif variant == 3:
            pts = [(0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(3,2)]
        else:
            pts = [(0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,1),(1,2),(2,1),(2,2),(3,4)]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # Configuration 4: Highly symmetric - use a 3x4 grid but with specific values
    # Try: x in {0,1,3,4}, y in {0,1,3}
    for xs, ys in [
        ([0,1,3,4], [0,1,3]),
        ([0,1,2,4], [0,1,3]),
        ([0,1,3,5], [0,1,3]),
        ([0,1,2,5], [0,1,3]),
        ([0,1,3,4], [0,1,4]),
        ([0,1,2,4], [0,1,4]),
    ]:
        if time.time() > deadline:
            break
        pts = [(x, y) for x in xs for y in ys]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # Configuration 5: Use points from a finite field construction
    # F_13 has 12 nonzero elements, but let's try a different approach
    # Use 12 points: (i, i^2 mod 13) for i=1..12, scaled to integers
    for scale in [1, 2, 5]:
        if time.time() > deadline:
            break
        pts = [(i*scale, (i*i % 13)*scale) for i in range(1, 13)]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # Configuration 6: Böröczky-like - 6 points on a conic, 6 at "infinity"
    # Use: (t, t^2) for t in {0,1,2,3,4,5} and (1/t, 1/t^2) approximated
    # Better: use (t, t^2) and (-t, t^2) for t=1..3, plus (0,0) and (0,1)
    for tset in [[1,2,3], [1,2,4], [1,3,5], [2,3,4]]:
        if time.time() > deadline:
            break
        pts = [(0, 0), (0, 1)]
        for t in tset:
            pts.append((t, t*t))
            pts.append((-t, t*t))
        # Need 12 points: 2 + 2*3 = 8, need 4 more
        # Add (1, 4), (-1, 4), (2, 9), (-2, 9) if not already there
        existing = set(tuple(p) for p in pts)
        for extra in [(1, 4), (-1, 4), (2, 9), (-2, 9), (1, 9), (-1, 9), (3, 9), (-3, 9)]:
            if len(pts) >= 12:
                break
            if tuple(extra) not in existing:
                pts.append(list(extra))
                existing.add(tuple(extra))
        if len(pts) == 12 and _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # Configuration 7: 12 points as vertices of a regular dodecagon approximated by integers
    # Use (cos(2πk/12), sin(2πk/12)) scaled - but these aren't integers
    # Instead use a known integer approximation
    
    # Configuration 8: Try the "6+6" with one group on a line and the other on a parabola
    for a in range(1, 4):
        if time.time() > deadline:
            break
        pts = [(i, 0) for i in range(6)]  # 6 on x-axis
        pts += [(i, a*i*i) for i in range(1, 7)]  # 6 on parabola y=a*x^2
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count == 6:
                return best_points
    
    # If we found something, return it
    if best_points is not None:
        return best_points
    
    # Fallback: return the original grid
    return [[x, y] for x in range(4) for y in range(3)]
# EVOLVE-BLOCK-END
