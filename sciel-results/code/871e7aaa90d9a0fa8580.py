"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = 150.0  # wall-clock kill limit; we guard with margin

def _elapsed():
    return time.time() - _START_TIME

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use a set of normalized line representations
    # A line through (x1,y1) and (x2,y2) can be represented by (A,B,C) where
    # A*x + B*y + C = 0, normalized so that gcd(A,B,C)=1 and A>0 or (A==0 and B>0)
    lines = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            # Normalize
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            # Ensure canonical sign: first non-zero of A,B,C is positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            lines.add((A, B, C))
    
    # Now count how many points are on each line
    ordinary = 0
    for (A, B, C) in lines:
        cnt = 0
        for (x, y) in points:
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    return ordinary

def _count_ordinary_lines_fast(points):
    """Faster count using slope grouping from each point."""
    n = len(points)
    if n < 2:
        return 0
    from math import gcd
    # For each point, group other points by the direction (slope) from it.
    # A line with k points through point i means k-1 other points share the same slope from i.
    # But we need to count each line exactly once. Better approach:
    # For each pair, compute the line. Then for each line, count points on it.
    # This is O(n^3) in worst case but n=12 so it's fine.
    
    # Actually for n=12, the naive O(n^3) is fine: 12 choose 2 = 66 pairs, each checks 12 points.
    return _count_ordinary_lines(points)

def _not_all_collinear(points):
    """Check that points are not all on one line."""
    if len(points) < 3:
        return True
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    # Cross product
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) != 0

def _validate(points):
    """Check all constraints."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
    if not _not_all_collinear(points):
        return False
    return True

def _initial_config():
    """Start with a 3x4 grid which has a known number of ordinary lines."""
    # 3x4 grid: points (x,y) for x in 0..3, y in 0..2
    pts = [[x, y] for x in range(4) for y in range(3)]
    return pts

def _local_search(points, max_iter=50000, time_limit=None):
    """Hill-climbing: try moving each point to a nearby integer position."""
    best = list(points)
    best_score = _count_ordinary_lines_fast(best)
    n = len(best)
    
    # Precompute for speed
    for iteration in range(max_iter):
        if time_limit is not None and _elapsed() > time_limit:
            break
        improved = False
        # Try moving each point to a small neighborhood
        for i in range(n):
            x0, y0 = best[i]
            # Try a few candidate moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check distinctness
                    new_pts = list(best)
                    new_pts[i] = [nx, ny]
                    if len(set(tuple(p) for p in new_pts)) != 12:
                        continue
                    if not _not_all_collinear(new_pts):
                        continue
                    score = _count_ordinary_lines_fast(new_pts)
                    if score < best_score:
                        best = new_pts
                        best_score = score
                        improved = True
                        # Reset x0,y0 to new position
                        x0, y0 = nx, ny
                        break
                else:
                    continue
                break
            if not improved:
                # Also try random jumps occasionally
                if iteration % 100 == 0:
                    for _ in range(5):
                        i = random.randint(0, n-1)
                        x0, y0 = best[i]
                        dx = random.randint(-10, 10)
                        dy = random.randint(-10, 10)
                        nx, ny = x0 + dx, y0 + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pts = list(best)
                        new_pts[i] = [nx, ny]
                        if len(set(tuple(p) for p in new_pts)) != 12:
                            continue
                        if not _not_all_collinear(new_pts):
                            continue
                        score = _count_ordinary_lines_fast(new_pts)
                        if score < best_score:
                            best = new_pts
                            best_score = score
                            improved = True
                            break
        if not improved and iteration > 0 and iteration % 500 == 0:
            # Stuck; try a bigger random perturbation
            i = random.randint(0, n-1)
            x0, y0 = best[i]
            for _ in range(20):
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                nx, ny = x0 + dx, y0 + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pts = list(best)
                new_pts[i] = [nx, ny]
                if len(set(tuple(p) for p in new_pts)) != 12:
                    continue
                if not _not_all_collinear(new_pts):
                    continue
                score = _count_ordinary_lines_fast(new_pts)
                if score < best_score:
                    best = new_pts
                    best_score = score
                    break
    return best, best_score

def _try_known_configs():
    """Try several hand-crafted or known good configurations."""
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 12 points on 3 lines of 4 (but not all collinear)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3
    # Line 3: y=2, x=0,1,2,3
    # That's just the 3x4 grid again.
    
    # Try: 3 lines with 4 points each, intersecting
    # Line 1: (0,0),(1,0),(2,0),(3,0)
    # Line 2: (0,0),(0,1),(0,2),(0,3)
    # Line 3: (0,0),(1,1),(2,2),(3,3)
    # But (0,0) is shared, so we have 3+3+3 = 9 unique points. Need 12.
    # Let's do: 3 lines, each with 4 points, no shared points, but that's 12 points.
    # However, if they don't intersect, they're parallel or skew. In 2D, non-parallel lines intersect.
    # So 3 non-parallel lines in 2D must have intersection points.
    
    # Let's try a configuration based on the Hesse configuration or similar.
    # For 12 points, a good candidate might be related to the 4x3 grid or a perturbation.
    
    # Try: points on a circle with integer coords? Hard to get many collinear.
    # Actually we want to MINIMIZE ordinary lines, so we want MANY points collinear.
    
    # 4 lines of 3 points each? 4*3=12 if no overlaps. But in 2D, lines intersect.
    # If 4 lines are in general position, they have 6 intersection points.
    # This is getting complex. Let's just use the local search from the grid.
    
    return candidates

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_pts = None
    best_score = float('inf')
    
    # Try known configurations
    for cand in _try_known_configs():
        if _validate(cand):
            score = _count_ordinary_lines_fast(cand)
            if score < best_score:
                best_score = score
                best_pts = [list(p) for p in cand]
    
    # If no valid candidate, use the grid
    if best_pts is None:
        best_pts = _initial_config()
        best_score = _count_ordinary_lines_fast(best_pts)
    
    # Local search
    time_limit = _DEADLINE - 10  # 10 second safety margin
    best_pts, best_score = _local_search(best_pts, max_iter=100000, time_limit=time_limit)
    
    # Verify
    if not _validate(best_pts):
        # Fallback to grid
        best_pts = _initial_config()
    
    return best_pts
# EVOLVE-BLOCK-END
