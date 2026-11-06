import time
import math
import random
from collections import defaultdict

def _gcd3(a, b, c):
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g

def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    a = dy
    b = -dx
    c = dx * y1 - dy * x1
    g = _gcd3(abs(a), abs(b), abs(c))
    if g:
        a //= g
        b //= g
        c //= g
    # Normalize sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            lines[_line_key(x1, y1, x2, y2)].add(i)
    return sum(1 for s in lines.values() if len(s) == 2)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(map(tuple, points))) < 12:
        return False
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return False
    # Check not all collinear
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    for k in range(3, 12):
        xi, yi = points[k]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _build_pencil_12():
    """
    Constructs 12 points on 4 concurrent lines through the origin.
    Distribution: 3, 3, 3, 2 points on the rays (excluding origin).
    Origin is included. Total 1 + 3 + 3 + 3 + 2 = 12.
    Directions chosen to maximize accidental collinearities if possible, 
    or simply standard orthogonal/diagonal.
    """
    # Directions: (1,0), (0,1), (1,1), (1,-1)
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    counts = [3, 3, 3, 2] # Points on each ray, not counting origin
    
    pts = [[0, 0]] # Origin
    for i, (ux, uy) in enumerate(dirs):
        for k in range(1, counts[i] + 1):
            pts.append([k * ux, k * uy])
            
    return pts

def _build_grid_12():
    """
    3x4 Grid.
    """
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _build_pencil_12_variant2():
    """
    Try different directions to see if we can get more collinearities.
    """
    # Directions: (1,0), (0,1), (1,2), (2,1)
    dirs = [(1, 0), (0, 1), (1, 2), (2, 1)]
    counts = [3, 3, 3, 2]
    
    pts = [[0, 0]]
    for i, (ux, uy) in enumerate(dirs):
        for k in range(1, counts[i] + 1):
            pts.append([k * ux, k * uy])
            
    return pts

def construct():
    start = time.time()
    deadline = start + 115
    
    best_points = None
    best_score = float('inf')
    
    def consider(pts):
        nonlocal best_points, best_score
        if not _is_valid(pts):
            return
        s = count_ordinary_lines(pts)
        if s < best_score:
            best_score = s
            best_points = [[int(x), int(y)] for x, y in pts]
    
    # Try structural initializers
    consider(_build_pencil_12())
    consider(_build_grid_12())
    consider(_build_pencil_12_variant2())
    
    # If we already found the theoretical minimum (6), we can stop early
    # but we might find a configuration that is even better or just confirm 6.
    # The minimum for 12 points is 6.
    
    if best_score <= 6:
        return best_points

    # Local Search to improve the best candidate found
    rng = random.Random(42)
    cur = [p[:] for p in best_points]
    cur_score = best_score
    
    # Simple local search: move one point slightly
    # To speed up, we precompute lines? 
    # Given 120s limit, and 12 points, O(N^2) counting is fast (144 pairs).
    # We can do many iterations.
    
    def attempt_move(idx, new_pt):
        nonlocal cur, cur_score, best_points, best_score
        if new_pt is None:
            return
        # Check bounds
        if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
            return
        # Check duplicate
        if any(tuple(new_pt) == tuple(p) for p in cur):
            return
        
        old_pt = cur[idx]
        cur[idx] = new_pt
        
        if not _is_valid(cur):
            cur[idx] = old_pt
            return
            
        s = count_ordinary_lines(cur)
        
        if s < cur_score:
            cur_score = s
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cur]
        else:
            # Revert
            cur[idx] = old_pt

    iters = 0
    while time.time() < deadline and iters < 500000:
        iters += 1
        idx = rng.randint(0, 11)
        px, py = cur[idx]
        
        # Random small move
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        attempt_move(idx, [px + dx, py + dy])
        
        # Try snapping to lines defined by other points
        # Pick two other points and try to place idx on the line through them
        # or on a line parallel to them?
        # Simpler: try to align with a random existing line segment
        if iters % 10 == 0:
            a, b = rng.sample(range(12), 2)
            if a != idx and b != idx:
                p1, p2 = cur[a], cur[b]
                # Try to place idx on the line p1-p2 extended
                # Vector
                vx = p2[0] - p1[0]
                vy = p2[1] - p1[1]
                g = math.gcd(abs(vx), abs(vy))
                if g > 0:
                    vx //= g
                    vy //= g
                    # Try t = -2, -1, 1, 2 relative to p1
                    for t in [-2, -1, 1, 2]:
                        nx = p1[0] + t * vx
                        ny = p1[1] + t * vy
                        attempt_move(idx, [nx, ny])

    if best_points is None:
        best_points = _build_grid_12()
        
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(count_ordinary_lines(pts))
