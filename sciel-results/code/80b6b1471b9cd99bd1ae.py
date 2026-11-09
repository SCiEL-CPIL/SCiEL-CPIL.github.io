"""Search for integral point sets: points with rational coordinates and rational pairwise distances,
no three collinear, no four concyclic."""
import math
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    """Check if n is a perfect square (n must be non-negative integer)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_pairwise_distances(points):
    """Check all pairwise distances are rational (squared distance is perfect square for D=1).
    Returns True if all distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact integer arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                # Cross product (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    """Check no four points are concyclic using exact integer arithmetic.
    Four points (ax,ay),(bx,by),(cx,cy),(dx,dy) are concyclic iff the determinant is 0."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    dx, dy = points[l]
                    # Circle through 3 points: determinant of augmented matrix
                    # |ax ay ax^2+ay^2 1|
                    # |bx by bx^2+by^2 1| = 0
                    # |cx cy cx^2+cy^2 1|
                    # |dx dy dx^2+dy^2 1|
                    a1 = ax * ax + ay * ay
                    b1 = bx * bx + by * by
                    c1 = cx * cx + cy * cy
                    d1 = dx * dx + dy * dy
                    # 4x4 determinant expansion
                    # Using the formula: det = ax*(by*(c1-d1) - cy*(b1-d1) + (by*cy - cy*by)*(0)) ... 
                    # Actually let's use a cleaner formula.
                    # The circle equation: x^2+y^2+Dx+Ey+F=0
                    # For point (x,y): x^2+y^2+Dx+Ey+F=0
                    # Subtract first equation from others to eliminate x^2+y^2:
                    # (bx-ax)D + (by-ay)E = ax^2+ay^2 - bx^2-by^2
                    # (cx-ax)D + (cy-ay)E = ax^2+ay^2 - cx^2-cy^2
                    # (dx-ax)D + (dy-ay)E = ax^2+ay^2 - dx^2-dy^2
                    # This is a 3x3 system. Points concyclic iff determinant of this system is 0.
                    # Actually simpler: compute the 4x4 determinant directly.
                    # det = ax*(by*c1 - by*d1 - cy*b1 + cy*d1 + (by*cy-cy*by)*0) ... let me just compute it.
                    
                    # Using cofactor expansion along last column (all 1s):
                    # det = 1*M14 - 1*M24 + 1*M34 - 1*M44 where Mij is minor
                    # M14 = det of [[bx,by,b1],[cx,cy,c1],[dx,dy,d1]]
                    # M24 = det of [[ax,ay,a1],[cx,cy,c1],[dx,dy,d1]]
                    # M34 = det of [[ax,ay,a1],[bx,by,b1],[dx,dy,d1]]
                    # M44 = det of [[ax,ay,a1],[bx,by,b1],[cx,cy,c1]]
                    
                    def det3(p1, p2, p3, q1, q2, q3, r1, r2, r3):
                        return (p1*(q2*r3 - q3*r2) - q1*(p2*r3 - p3*r2) + r1*(p2*q3 - p3*q2))
                    
                    M14 = det3(bx, by, b1, cx, cy, c1, dx, dy, d1)
                    M24 = det3(ax, ay, a1, cx, cy, c1, dx, dy, d1)
                    M34 = det3(ax, ay, a1, bx, by, b1, dx, dy, d1)
                    M44 = det3(ax, ay, a1, bx, by, b1, cx, cy, c1)
                    
                    det_val = M14 - M24 + M34 - M44
                    if det_val == 0:
                        return False
    return True

def _check_valid(points):
    """Full validity check: pairwise rational distances, no 3 collinear, no 4 concyclic."""
    if not _check_pairwise_distances(points):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points):
        return False
    return True

def _find_candidates_with_dist(x, y, max_coord, existing_points):
    """Find integer points (x,y) such that distance to all existing points is integer."""
    candidates = []
    for nx in range(-max_coord, max_coord + 1):
        for ny in range(-max_coord, max_coord + 1):
            if nx == x and ny == y:
                continue
            ok = True
            for (ex, ey) in existing_points:
                dx = nx - ex
                dy = ny - ey
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    ok = False
                    break
            if ok:
                candidates.append((nx, ny))
    return candidates

def _search_incremental(start_points, max_coord, deadline, max_points_target=8):
    """Incrementally search for larger valid integral point sets."""
    best = list(start_points)
    best_size = len(best)
    
    # Sort by size, try to extend
    current = list(start_points)
    current_size = len(current)
    
    # Try to extend one point at a time
    while current_size < max_points_target and time.time() < deadline:
        # Find all candidate points that maintain validity
        # We need a point that has integer distance to ALL current points
        # and doesn't create collinearity or concyclicity
        
        candidates = _find_candidates_with_dist(0, 0, max_coord, current)
        
        # Filter candidates: no three collinear with any two existing, no four concyclic
        valid_candidates = []
        for (cx, cy) in candidates:
            # Check no three collinear
            collinear = False
            for i in range(current_size):
                for j in range(i + 1, current_size):
                    ax, ay = current[i]
                    bx, by = current[j]
                    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                    if cross == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check no four concyclic with any three existing
            concyclic = False
            for i in range(current_size):
                for j in range(i + 1, current_size):
                    for k in range(j + 1, current_size):
                        ax, ay = current[i]
                        bx, by = current[j]
                        ex, ey = current[k]
                        fx, fy = (cx, cy)
                        a1 = ax * ax + ay * ay
                        b1 = bx * bx + by * by
                        e1 = ex * ex + ey * ey
                        f1 = fx * fx + fy * fy
                        
                        def det3(p1, p2, p3, q1, q2, q3, r1, r2, r3):
                            return (p1*(q2*r3 - q3*r2) - q1*(p2*r3 - p3*r2) + r1*(p2*q3 - p3*q2))
                        
                        M14 = det3(bx, by, b1, ex, ey, e1, fx, fy, f1)
                        M24 = det3(ax, ay, a1, ex, ey, e1, fx, fy, f1)
                        M34 = det3(ax, ay, a1, bx, by, b1, fx, fy, f1)
                        M44 = det3(ax, ay, a1, bx, by, b1, ex, ey, e1)
                        
                        det_val = M14 - M24 + M34 - M44
                        if det_val == 0:
                            concyclic = True
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
            
            valid_candidates.append((cx, cy))
        
        if not valid_candidates:
            break
        
        # Try each candidate, pick the one that allows further extension
        best_extension = None
        best_future_size = 0
        
        for (cx, cy) in valid_candidates:
            if time.time() >= deadline:
                break
            new_current = current + [(cx, cy)]
            # Check if we can extend further
            future_candidates = _find_candidates_with_dist(0, 0, max_coord, new_current)
            # Quick filter
            fc_count = 0
            for (fx, fy) in future_candidates:
                # Just count, don't do full check yet for speed
                fc_count += 1
                if fc_count > best_future_size:
                    break
            
            if fc_count > best_future_size or (best_extension is None):
                best_extension = (cx, cy)
                best_future_size = fc_count
        
        if best_extension is None:
            break
        
        current.append(best_extension)
        current_size += 1
        
        if current_size > best_size:
            best = list(current)
            best_size = current_size
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a known valid 5-point set
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    start_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the starting set is valid
    assert _check_valid(start_points), "Starting set is not valid!"
    
    best = list(start_points)
    best_size = len(best)
    
    # Try incremental extension with increasing search radius
    for max_coord in [100, 200, 300, 400, 500]:
        if time.time() >= deadline - 5:
            break
        result = _search_incremental(start_points, max_coord, deadline, max_points_target=8)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Also try different starting configurations
    # Try a few other known small integral sets
    alt_starts = [
        [(0, 0), (3, 4), (5, 0), (0, 12), (9, 12)],
        [(0, 0), (7, 24), (15, 8), (0, 24), (8, 24)],
    ]
    
    for alt in alt_starts:
        if time.time() >= deadline - 5:
            break
        if _check_valid(alt):
            for max_coord in [100, 200, 300]:
                if time.time() >= deadline - 5:
                    break
                result = _search_incremental(alt, max_coord, deadline, max_points_target=8)
                if len(result) > best_size:
                    best = result
                    best_size = len(result)
    
    # Convert to required format
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': 1, 'points': points_out}

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result['points'])} points")
    for p in result['points']:
        print(f"  ({p[0][0]}/{p[0][1]}, {p[1][0]}/{p[1][1]})")
