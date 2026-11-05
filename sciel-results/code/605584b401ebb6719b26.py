import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 140.0  # 10s safety margin
_SAFETY = 5.0
_DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_4lines_and_check(points):
    """
    Returns (count_of_4lines, is_valid).
    is_valid is False if any 5 points are collinear or points are out of bounds/duplicate.
    """
    n = len(points)
    if n != 16:
        return 0, False
    
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
        if (x, y) in seen:
            return 0, False
        seen.add((x, y))
        
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx, dy = x2 - x1, y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g:
                dx, dy = dx // g, dy // g
            else:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            
            # Line equation: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            if key in line_points:
                line_points[key].add(i)
                line_points[key].add(j)
            else:
                line_points[key] = {i, j}
                
    count = 0
    for pts_on_line in line_points.values():
        if len(pts_on_line) >= 5:
            return 0, False
        if len(pts_on_line) == 4:
            count += 1
            
    return count, True

def _score(points):
    c, valid = _count_4lines_and_check(points)
    return c if valid else -1

def _grid44():
    return [[x, y] for x in range(4) for y in range(4)]

def _random_points(rng):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append([x, y])
    return pts

def _get_3_point_lines(points):
    """
    Returns a list of tuples: (line_key, set_of_3_indices)
    for all lines containing exactly 3 points.
    """
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx, dy = x2 - x1, y2 - y1
            g = _gcd(abs(dx), abs(dy))
            if g:
                dx, dy = dx // g, dy // g
            else:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            if key in line_points:
                line_points[key].add(i)
                line_points[key].add(j)
            else:
                line_points[key] = {i, j}
                
    lines_3 = []
    for key, pts in line_points.items():
        if len(pts) == 3:
            lines_3.append((key, pts))
    return lines_3

def _local_search(points, rng, max_iters=200):
    """
    Local search with heuristic moves.
    """
    pts = [list(p) for p in points]
    best = [list(p) for p in pts]
    best_score = _score(best)
    if best_score < 0:
        best_score = 0 # invalid, but we handle it in construct

    for it in range(max_iters):
        if time.time() > _DEADLINE_EFF:
            break
        
        cur_score = _score(pts)
        if cur_score < 0:
            # If invalid, try to fix by moving a random point
            i = rng.randrange(16)
            old = list(pts[i])
            radius = rng.choice([1, 2, 5, 10])
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            nx = pts[i][0] + dx
            ny = pts[i][1] + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                dup = False
                for k in range(16):
                    if k != i and pts[k][0] == nx and pts[k][1] == ny:
                        dup = True
                        break
                if not dup:
                    pts[i] = [nx, ny]
                    # Don't update best if still invalid, but keep searching
                    continue
            else:
                pts[i] = old
                continue

        # Try heuristic move: complete a 3-point line
        lines_3 = _get_3_point_lines(pts)
        move_made = False
        if lines_3:
            # Pick a random 3-point line
            key, three_pts = rng.choice(lines_3)
            # The 3 points are at indices in three_pts
            # We want to move one of the OTHER 13 points to a 4th position on this line
            # Or move one of the 3 points? No, moving one of the 3 points breaks the line.
            # We must move a point NOT in three_pts to a new position on this line.
            
            # Find a candidate position on the line (dx, dy, c)
            # Line: dy*x - dx*y = c
            # We need an integer solution (x, y) close to some existing point?
            # Or just any integer solution in bounds?
            # Let's pick a random existing point P_k (k not in three_pts) and try to move it to a point on the line.
            
            available_indices = [i for i in range(16) if i not in three_pts]
            if available_indices:
                k = rng.choice(available_indices)
                # Find an integer point (nx, ny) on the line dy*nx - dx*ny = c
                # that is close to pts[k] or just random in bounds?
                # Let's try to find a point on the line within a small radius of pts[k]
                found = False
                for r in range(1, 10):
                    if found: break
                    for dx_ in range(-r, r+1):
                        if found: break
                        for dy_ in range(-r, r+1):
                            nx = pts[k][0] + dx_
                            ny = pts[k][1] + dy_
                            if abs(nx) > 1000 or abs(ny) > 1000:
                                continue
                            # Check if on line
                            if dy * nx - dx * ny == c:
                                # Check if it's already one of the 3 points
                                if (nx, ny) in [(pts[i][0], pts[i][1]) for i in three_pts]:
                                    continue
                                # Check if it collides with any other point
                                dup = False
                                for m in range(16):
                                    if m != k and pts[m][0] == nx and pts[m][1] == ny:
                                        dup = True
                                        break
                                if not dup:
                                    # Try the move
                                    old_k = list(pts[k])
                                    pts[k] = [nx, ny]
                                    new_score = _score(pts)
                                    if new_score > best_score:
                                        best_score = new_score
                                        best = [list(p) for p in pts]
                                        move_made = True
                                    else:
                                        pts[k] = old_k
                                    found = True
                                    break

        if not move_made:
            # Try random move
            i = rng.randrange(16)
            old = list(pts[i])
            radius = rng.choice([1, 2, 3, 5])
            dx = rng.randint(-radius, radius)
            dy = rng.randint(-radius, radius)
            nx = pts[i][0] + dx
            ny = pts[i][1] + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                dup = False
                for k in range(16):
                    if k != i and pts[k][0] == nx and pts[k][1] == ny:
                        dup = True
                        break
                if not dup:
                    pts[i] = [nx, ny]
                    new_score = _score(pts)
                    if new_score > best_score:
                        best_score = new_score
                        best = [list(p) for p in pts]
                    else:
                        pts[i] = old
                    move_made = True

    return best, best_score

def construct():
    global _START_TIME, _DEADLINE_EFF
    _START_TIME = time.time()
    _DEADLINE_EFF = _START_TIME + _DEADLINE - _SAFETY

    best_pts = _grid44()
    best_score = _score(best_pts)

    rng = random.Random(42)

    # Phase 1: Local search on grid
    best_pts, best_score = _local_search(best_pts, rng, max_iters=500)

    # Phase 2: Random restarts
    while time.time() < _DEADLINE_EFF - 10:
        # Generate random start
        pts = _random_points(rng)
        # Run local search
        p, s = _local_search(pts, rng, max_iters=200)
        if s > best_score:
            best_score = s
            best_pts = p

    # Phase 3: Final polish
    if time.time() < _DEADLINE_EFF - 5:
        best_pts, best_score = _local_search(best_pts, rng, max_iters=1000)

    # Final check
    if _score(best_pts) < 0:
        # Fallback to grid if something went wrong
        best_pts = _grid44()
        
    return [[int(x), int(y)] for x, y in best_pts]
