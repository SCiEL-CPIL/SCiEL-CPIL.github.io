"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from collections import Counter

# EVOLVE-BLOCK-START
def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line key: (slope_dx, slope_dy, intercept) where intercept = y*dx - x*dy (for normalized direction)
    # Actually use: a*x + b*y = c where a=dy, b=-dx, c=dy*x1 - dx*y1, normalized by gcd
    a = dy
    b = -dx
    c = a * x1 + b * y1
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_lines(points):
    """Count lines with exactly 4 points and check no 5 collinear.
    Returns (num_exact4, has_five) where has_five is True if any line has >=5 points."""
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] += 1
    # Each line with k points appears C(k,2) times
    # We need to find actual number of points per line
    # Reconstruct: for each unique line key, count how many points lie on it
    # Actually, from the pair counting, if a line has k points, it's counted C(k,2) times.
    # So we can't directly get k from the counter value.
    # Better approach: for each line key, count points on it directly.
    
    # More efficient: group points by line
    # For each pair, we get a line key. A line with k points has C(k,2) pairs.
    # We need to find k for each line.
    # Alternative: just check all lines defined by pairs, count points on each.
    
    # Let's do it differently: collect all unique lines, then for each, count points.
    unique_lines = set(line_counts.keys())
    exact4 = 0
    has_five = False
    for key in unique_lines:
        a, b, c = key
        cnt = 0
        for pt in points:
            if a * pt[0] + b * pt[1] == c:
                cnt += 1
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            has_five = True
    return exact4, has_five

def _score(points):
    """Return score: number of exact-4 lines, or -1 if invalid (5+ collinear)."""
    exact4, has_five = _count_lines(points)
    if has_five:
        return -1
    return exact4

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(points)
    best_points = [p[:] for p in points]
    
    # Local search: try moving each point to nearby positions
    # Search neighborhood: try displacing one point by (dx, dy) in a range
    step_sizes = list(range(1, 11))  # Try displacements up to 10
    
    while time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            orig = points[i][:]
            # Try moving point i to various nearby positions
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    new_pt = [nx, ny]
                    if any(new_pt == points[j] for j in range(16)):
                        continue
                    # Try the move
                    old_pt = points[i]
                    points[i] = new_pt
                    sc = _score(points)
                    if sc > best_score:
                        best_score = sc
                        best_points = [p[:] for p in points]
                        improved = True
                    else:
                        points[i] = old_pt
        if not improved:
            # Try larger steps
            for i in range(16):
                if time.time() > deadline:
                    break
                orig = points[i][:]
                for dx in range(-30, 31, 2):
                    for dy in range(-30, 31, 2):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        new_pt = [nx, ny]
                        if any(new_pt == points[j] for j in range(16)):
                            continue
                        old_pt = points[i]
                        points[i] = new_pt
                        sc = _score(points)
                        if sc > best_score:
                            best_score = sc
                            best_points = [p[:] for p in points]
                            improved = True
                        else:
                            points[i] = old_pt
                if improved:
                    break
            if not improved:
                # Try even larger random perturbations
                import random
                random.seed(42)
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    i = random.randint(0, 15)
                    orig = points[i][:]
                    dx = random.randint(-50, 50)
                    dy = random.randint(-50, 50)
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = [nx, ny]
                    if any(new_pt == points[j] for j in range(16)):
                        continue
                    old_pt = points[i]
                    points[i] = new_pt
                    sc = _score(points)
                    if sc > best_score:
                        best_score = sc
                        best_points = [p[:] for p in points]
                        improved = True
                    else:
                        points[i] = old_pt
                if not improved:
                    break
    
    # Return best points found
    return [p[:] for p in best_points]
# EVOLVE-BLOCK-END
