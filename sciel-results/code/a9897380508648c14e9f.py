"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize: make dx positive, or if dx==0 make dy positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Compute gcd to reduce
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Line: dy*x - dx*y = c where c = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_points_on_line(line_key, points):
    """Count how many points lie on the given line."""
    if line_key is None:
        return 0
    dx, dy, c = line_key
    count = 0
    for x, y in points:
        if dy * x - dx * y == c:
            count += 1
    return count

def _score(points):
    """Count number of lines containing exactly 4 of the points."""
    n = len(points)
    if n != 16:
        return -1
    # Check for 5+ collinear
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = _count_points_on_line(key, points)
            # Early check for 5+
            if line_counts[key] >= 5:
                return -1
    count4 = 0
    for key, cnt in line_counts.items():
        if cnt == 4:
            count4 += 1
    return count4

def _is_valid(points):
    """Check if points are valid: 16 distinct, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check no 5 collinear
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = _count_points_on_line(key, points)
            if line_counts[key] >= 5:
                return False
    return True

def _local_search(points, deadline):
    """Improve points via local search."""
    best = [list(p) for p in points]
    best_score = _score(best)
    if best_score < 0:
        best_score = 0
    
    while time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() > deadline:
                break
            x, y = best[idx]
            # Try nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    new_points = [list(p) for p in best]
                    new_points[idx] = [nx, ny]
                    # Quick check for duplicates
                    if len(set(tuple(p) for p in new_points)) != 16:
                        continue
                    s = _score(new_points)
                    if s > best_score:
                        best = new_points
                        best_score = s
                        improved = True
                        x, y = nx, ny
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try larger moves
            for _ in range(50):
                if time.time() > deadline:
                    break
                idx = random.randint(0, 15)
                x, y = best[idx]
                nx = x + random.randint(-20, 20)
                ny = y + random.randint(-20, 20)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_points = [list(p) for p in best]
                new_points[idx] = [nx, ny]
                if len(set(tuple(p) for p in new_points)) != 16:
                    continue
                s = _score(new_points)
                if s > best_score:
                    best = new_points
                    best_score = s
                    break
            else:
                # No improvement, try random restarts
                break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [list(p) for p in points]
    best_score = _score(best_points)
    
    # Local search
    best_points = _local_search(best_points, deadline)
    final_score = _score(best_points)
    
    if final_score < 0:
        # Fallback to original
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points

# EVOLVE-BLOCK-END
