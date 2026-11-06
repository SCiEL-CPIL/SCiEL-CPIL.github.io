import time
import random
from math import gcd
from collections import Counter

def _line_key(x1, y1, x2, y2):
    """Return a canonical key for the line through (x1,y1) and (x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # c = dy*x - dx*y evaluated at point 1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score(points):
    """
    Return (exact4_count, has_five) for a set of 16 points.
    exact4_count: number of lines containing exactly 4 of the points.
    has_five: True if any line contains >= 5 points.
    """
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    exact4 = 0
    has_five = False
    for key, pc in line_counts.items():
        # pc = C(k,2) where k is number of points on line
        if pc == 6:
            exact4 += 1
        elif pc >= 10:
            has_five = True
            break
    return exact4, has_five

def _fast_score(points):
    """
    Faster scoring using numpy-free approach with early termination.
    Returns (exact4_count, has_five).
    """
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    exact4 = 0
    has_five = False
    for key, pc in line_counts.items():
        if pc == 6:
            exact4 += 1
        elif pc >= 10:
            has_five = True
            break
    return exact4, has_five

def _find_lines(points):
    """
    Return dict: line_key -> list of point indices on that line.
    Only for lines with >= 2 points.
    """
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    # Convert sets to lists
    return {k: list(v) for k, v in line_points.items()}

def _is_valid(points):
    """Check that points are 16 distinct points within bounds and no 5 collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    if not all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in points):
        return False
    _, has_five = _fast_score(points)
    return not has_five

def _line_completion_moves(points, lines):
    """
    Generate candidate moves that complete a 4-point line.
    For each line with exactly 3 points, find the 4th point that would complete it.
    Also for lines with 2 points, consider adding 2 more (but that's harder).
    Focus on 3->4 completions.
    """
    moves = []
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 3:
            # Get the three points
            p1 = points[pts_on_line[0]]
            p2 = points[pts_on_line[1]]
            p3 = points[pts_on_line[2]]
            # The line is determined by any two; find the 4th point
            # that lies on this line at integer coordinates.
            # Use the line equation: dy*x - dx*y = c
            dx, dy, c = key
            # We need to find an integer point (x,y) on this line
            # that is NOT one of the three existing points,
            # and is within bounds, and not already used.
            # Strategy: try nearby points along the line direction.
            # The direction is (dx, dy). Points on the line can be parameterized.
            # Take p1 as base: (x1, y1) + t*(dx, dy) for integer t.
            x1, y1 = p1
            used_coords = set(tuple(p) for p in points)
            # Try t values that give integer coordinates within bounds
            # Since dx, dy are coprime (normalized), t can be any integer.
            # The three existing points correspond to specific t values.
            # Let's find the t values for the three points.
            t_values = []
            for p in [p1, p2, p3]:
                # (x - x1) / dx = (y - y1) / dy = t
                # Need to handle dx=0 or dy=0 cases
                if dx != 0:
                    if (p[0] - x1) % dx == 0:
                        t = (p[0] - x1) // dx
                        # Verify with dy
                        if dy != 0:
                            if (p[1] - y1) % dy != 0 or (p[1] - y1) // dy != t:
                                continue
                    else:
                        continue
                else:
                    # dx == 0, so dy != 0
                    if (p[1] - y1) % dy == 0:
                        t = (p[1] - y1) // dy
                    else:
                        continue
                t_values.append(t)
            
            # Now try t values near the existing ones
            for t in range(min(t_values) - 3, max(t_values) + 4):
                if t in t_values:
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in used_coords:
                    continue
                moves.append((nx, ny))
    
    # Deduplicate
    return list(set(moves))

def _local_search(points, deadline, max_no_improve=200):
    """
    Hill-climbing local search.
    Returns best points found and their score.
    """
    cur = [p[:] for p in points]
    cur_score, cur_five = _fast_score(cur)
    if cur_five:
        return cur, cur_score
    
    best = [p[:] for p in cur]
    best_score = cur_score
    no_improve = 0
    rng = random.Random(hash(tuple(tuple(p) for p in cur)) % 2**31)
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Get current lines
        lines = _find_lines(cur)
        
        # Generate candidate moves
        candidates = []
        
        # 1. Line completion moves
        comp_moves = _line_completion_moves(cur, lines)
        for (nx, ny) in comp_moves:
            candidates.append((nx, ny))
        
        # 2. Random perturbation of a single point
        for _ in range(8):
            idx = rng.randint(0, 15)
            dx = rng.randint(-6, 6)
            dy = rng.randint(-6, 6)
            nx = cur[idx][0] + dx
            ny = cur[idx][1] + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                candidates.append((nx, ny))
        
        # 3. Move a point that's on a 3-line to complete it (specific)
        # This is covered by line_completion_moves above
        
        found_improvement = False
        for (nx, ny) in candidates:
            if time.time() > deadline:
                break
            # Try moving each point to (nx, ny)
            # But we need to ensure (nx, ny) is not already occupied
            used = set(tuple(p) for p in cur)
            if (nx, ny) in used:
                continue
            
            # Try moving each point to this location
            for idx in range(16):
                new_pts = [p[:] for p in cur]
                new_pts[idx] = [nx, ny]
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
                s, f = _fast_score(new_pts)
                if f == 0 and s > cur_score:
                    cur = new_pts
                    cur_score = s
                    no_improve = 0
                    found_improvement = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best = [p[:] for p in cur]
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0
    
    # Initial candidate: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    s, f = _fast_score(grid)
    if f == 0:
        best_score = s
        best_points = [p[:] for p in grid]
    
    # Try several structured starting points
    starting_points = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid with different spacing
        [[x*2, y*2] for x in range(4) for y in range(4)],
        [[x*3, y*3] for x in range(4) for y in range(4)],
        # Perturbed 4x4 grids
        [[x + (y % 2), y] for x in range(4) for y in range(4)],
        [[x, y + (x % 2)] for x in range(4) for y in range(4)],
        [[x*2 + (y % 2), y*2] for x in range(4) for y in range(4)],
    ]
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        if len(set(tuple(p) for p in sp)) != 16:
            continue
        if not all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in sp):
            continue
        s, f = _fast_score(sp)
        if f == 0 and s > best_score:
            best_score = s
            best_points = [p[:] for p in sp]
    
    # Local search with multiple restarts
    rng = random.Random(12345)
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            cur = [p[:] for p in best_points]
        elif restart < 10:
            # Small perturbation of best
            cur = [p[:] for p in best_points]
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, 15)
                cur[idx][0] += rng.randint(-8, 8)
                cur[idx][1] += rng.randint(-8, 8)
                if abs(cur[idx][0]) > 1000:
                    cur[idx][0] = rng.randint(-500, 500)
                if abs(cur[idx][1]) > 1000:
                    cur[idx][1] = rng.randint(-500, 500)
        elif restart < 20:
            # Medium perturbation
            cur = [p[:] for p in best_points]
            for _ in range(rng.randint(3, 6)):
                idx = rng.randint(0, 15)
                cur[idx][0] += rng.randint(-20, 20)
                cur[idx][1] += rng.randint(-20, 20)
                if abs(cur[idx][0]) > 1000:
                    cur[idx][0] = rng.randint(-500, 500)
                if abs(cur[idx][1]) > 1000:
                    cur[idx][1] = rng.randint(-500, 500)
        else:
            # Random restart
            cur = [[rng.randint(-200, 200), rng.randint(-200, 200)] for _ in range(16)]
            # Ensure distinct
            for attempt in range(100):
                if len(set(tuple(p) for p in cur)) == 16:
                    break
                dup = rng.randint(0, 15)
                cur[dup][0] += rng.randint(-50, 50)
                cur[dup][1] += rng.randint(-50, 50)
        
        # Validate
        if len(set(tuple(p) for p in cur)) != 16:
            continue
        if not all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in cur):
            continue
        s, f = _fast_score(cur)
        if f:
            # Try to fix by moving points off 5-collinear lines
            # Simple: just skip this restart
            continue
        
        # Local search
        result, score = _local_search(cur, deadline, max_no_improve=150)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in result]
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s, f = _fast_score(final)
    assert f == 0, f"Five collinear found: {f}"
    
    return final
