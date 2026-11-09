import time
from math import gcd
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_map = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Line through (x1,y1) with direction (dx,dy): normal is (dy,-dx)
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    return sum(1 for s in line_map.values() if len(s) == 2)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    # Not all collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    for k in range(2, 12):
        xk, yk = points[k]
        if (y1 - y0) * (xk - x0) != (yk - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 110.0  # Safety margin

    # Generate diverse candidate configurations
    candidates = []

    # 3x4 grid (baseline: 8 ordinary lines)
    candidates.append([[x, y] for x in range(3) for y in range(4)])

    # 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])

    # Triangle with 4 on each edge (9 unique) + 3 interior
    candidates.append([[0,0],[1,0],[2,0],[3,0],[0,1],[0,2],[0,3],[1,2],[2,1],[1,1],[2,2],[1,3]])

    # Two parallel lines, 6 each
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])

    # 6 on x-axis, 6 on y-axis (cross)
    candidates.append([[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)])

    # Hexagonal arrangement
    candidates.append([[3,0],[0,3],[-3,0],[0,-3],[-3,-3],[3,-3],[1,2],[-1,2],[2,-1],[-2,-1],[1,-2],[-1,-2]])

    # 3 lines through origin with 4 points each (12 total, no shared origin)
    # y=0: (1,0),(2,0),(3,0),(4,0)
    # y=x: (1,1),(2,2),(3,3),(4,4)
    # y=2x: (1,2),(2,4),(3,6),(4,8)
    candidates.append([[1,0],[2,0],[3,0],[4,0],[1,1],[2,2],[3,3],[4,4],[1,2],[2,4],[3,6],[4,8]])

    # 4 lines through origin, 3 points each (12 total)
    # y=0: (1,0),(2,0),(3,0)
    # y=x: (1,1),(2,2),(3,3)
    # y=2x: (1,2),(2,4),(3,6)
    # y=3x: (1,3),(2,6),(3,9)
    candidates.append([[1,0],[2,0],[3,0],[1,1],[2,2],[3,3],[1,2],[2,4],[3,6],[1,3],[2,6],[3,9]])

    # Complete quadrangle: 4 points in general position, 6 intersection lines
    # Use points: (0,0), (1,0), (0,1), (1,1) - square
    # Plus 8 more points on the 6 lines connecting pairs
    # Lines: y=0, x=0, y=1, x=1, y=x, y=1-x
    # Add points: (2,0), (0,2), (2,1), (1,2), (2,2), (3,0), (0,3), (2,3)
    candidates.append([[0,0],[1,0],[0,1],[1,1],[2,0],[0,2],[2,1],[1,2],[2,2],[3,0],[0,3],[2,3]])

    # Affine plane AG(2,3) minus 3 points: 9 points in 3x3 grid, add 3 more
    # 3x3 grid: (0,0)-(2,2)
    # Add: (3,0), (0,3), (3,3)
    candidates.append([[x,y] for x in range(3) for y in range(3)] + [[3,0],[0,3],[3,3]])

    # Evaluate all candidates
    best_pts = None
    best_count = float('inf')
    for pts in candidates:
        if not _is_valid(pts):
            continue
        cnt = _count_ordinary_lines(pts)
        if cnt < best_count:
            best_count = cnt
            best_pts = pts[:]

    if best_pts is None:
        best_pts = [[x, y] for x in range(3) for y in range(4)]
        best_count = _count_ordinary_lines(best_pts)

    # Local search: try moving points to create new collinearities
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            x_orig, y_orig = best_pts[i]
            # Try a set of candidate positions that might create collinearities
            # First: try positions on lines through pairs of other points
            tried = set()
            for j in range(12):
                if j == i:
                    continue
                for k in range(j + 1, 12):
                    if k == i:
                        continue
                    xj, yj = best_pts[j]
                    xk, yk = best_pts[k]
                    dx = xk - xj
                    dy = yk - yj
                    # Points on line through j,k: (xj + t*dx, yj + t*dy)
                    for t in range(-3, 4):
                        nx = xj + t * dx
                        ny = yj + t * dy
                        if (nx, ny) == (x_orig, y_orig):
                            continue
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        if (nx, ny) in tried:
                            continue
                        tried.add((nx, ny))
                        new_pts = best_pts[:]
                        new_pts[i] = [nx, ny]
                        if not _is_valid(new_pts):
                            continue
                        cnt = _count_ordinary_lines(new_pts)
                        if cnt < best_count:
                            best_count = cnt
                            best_pts = new_pts
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

            # Also try small perturbations
            for dx in range(-5, 6):
                if time.time() > deadline:
                    break
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x_orig + dx
                    ny = y_orig + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    if (nx, ny) in tried:
                        continue
                    tried.add((nx, ny))
                    new_pts = best_pts[:]
                    new_pts[i] = [nx, ny]
                    if not _is_valid(new_pts):
                        continue
                    cnt = _count_ordinary_lines(new_pts)
                    if cnt < best_count:
                        best_count = cnt
                        best_pts = new_pts
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_pts
