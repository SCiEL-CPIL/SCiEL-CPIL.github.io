"""Minimize ordinary lines for 12 integer points via local search."""
import time
import random
import math
from collections import defaultdict

DEADLINE = time.time() + 135  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _build_line_map(points):
    """Build dict: line_key -> set of point indices on that line."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _count_ordinary_from_line_map(line_map):
    """Count ordinary lines from a line_map (line_key -> set of indices)."""
    return sum(1 for s in line_map.values() if len(s) == 2)

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _clamp(p, limit=10**9):
    x, y = p
    x = max(-limit, min(limit, x))
    y = max(-limit, min(limit, y))
    return [x, y]

def _fast_move(points, idx, new_pt, line_map):
    """
    Try moving point idx to new_pt. Returns (new_score, updated_line_map) or (None, None).
    Makes a copy of line_map to avoid mutation on rejection.
    """
    n = len(points)
    old_pt = points[idx]
    if new_pt == old_pt:
        return None, None
    # Check distinctness
    for i in range(n):
        if i != idx and points[i] == new_pt:
            return None, None
    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
        return None, None
    
    # Copy line_map
    new_line_map = {k: set(s) for k, s in line_map.items()}
    
    # Collect all lines involving point idx
    affected_lines = set()
    for j in range(n):
        if j != idx:
            key = _line_key(old_pt, points[j])
            affected_lines.add(key)
    
    # Remove idx from those lines
    for key in affected_lines:
        if key in new_line_map:
            new_line_map[key].discard(idx)
            if not new_line_map[key]:
                del new_line_map[key]
    
    # Add new lines
    for j in range(n):
        if j != idx:
            key = _line_key(new_pt, points[j])
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(idx)
            new_line_map[key].add(j)
    
    # Count ordinary lines
    score = sum(1 for s in new_line_map.values() if len(s) == 2)
    return score, new_line_map

def _fast_two_point_move(points, i1, i2, new1, new2, line_map):
    """
    Try moving points i1 and i2 to new1 and new2 simultaneously.
    Returns (new_score, updated_line_map) or (None, None).
    """
    n = len(points)
    old1 = points[i1]
    old2 = points[i2]
    
    # Check distinctness
    for i in range(n):
        if i not in (i1, i2):
            if points[i] == new1 or points[i] == new2:
                return None, None
    if new1 == new2:
        return None, None
    if abs(new1[0]) > 10**9 or abs(new1[1]) > 10**9:
        return None, None
    if abs(new2[0]) > 10**9 or abs(new2[1]) > 10**9:
        return None, None
    
    # Copy line_map
    new_line_map = {k: set(s) for k, s in line_map.items()}
    
    # Collect all lines involving i1 or i2
    affected_lines = set()
    for j in range(n):
        if j != i1:
            affected_lines.add(_line_key(old1, points[j]))
        if j != i2:
            affected_lines.add(_line_key(old2, points[j]))
    
    # Remove i1 and i2 from those lines
    for key in affected_lines:
        if key in new_line_map:
            new_line_map[key].discard(i1)
            new_line_map[key].discard(i2)
            if not new_line_map[key]:
                del new_line_map[key]
    
    # Add new lines for i1
    for j in range(n):
        if j != i1:
            key = _line_key(new1, points[j] if j != i2 else new2)
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(i1)
            if j != i2:
                new_line_map[key].add(j)
            else:
                new_line_map[key].add(i2)
    
    # Add new lines for i2
    for j in range(n):
        if j != i2:
            key = _line_key(new2, points[j] if j != i1 else new1)
            if key not in new_line_map:
                new_line_map[key] = set()
            new_line_map[key].add(i2)
            if j != i1:
                new_line_map[key].add(j)
            else:
                new_line_map[key].add(i1)
    
    # Count ordinary lines
    score = sum(1 for s in new_line_map.values() if len(s) == 2)
    return score, new_line_map

def construct():
    random.seed(42)
    
    # Expanded structured seeds based on known extremal configurations
    structured_seeds = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two lines
        [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)],
        # Points on three parallel lines
        [[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)],
        # Hexagon-like
        [[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
         [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]],
        # 12 points on a circle (approximate with integers)
        [[7, 24], [15, 8], [8, 15], [-7, 24], [-15, 8], [-8, 15],
         [7, -24], [15, -8], [8, -15], [-7, -24], [-15, -8], [-8, -15]],
        # Böröczky-like: 4 points on a line, 4 on another, 4 on a third
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 3], [1, 3], [2, 3], [3, 3]],
        # Points forming many diagonal lines
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 1], [1, 1], [2, 1], [3, 1],
         [0, 2], [1, 2], [2, 2], [3, 2]],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # Points on 4 lines of 3
        [[0, 0], [1, 0], [2, 0],
         [0, 1], [1, 1], [2, 1],
         [0, 2], [1, 2], [2, 2],
         [0, 3], [1, 3], [2, 3]],
        # 3x3 grid plus 3 extra points
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2],
         [3, 0], [3, 1], [3, 2]],
        # Symmetric cross pattern
        [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
         [1, -1], [-1, 1], [2, 0], [0, 2], [0, -2]],
        # 5 on line y=0, 5 on line y=1, 2 connecting
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
         [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],
         [5, 0], [5, 1]],
        # 4 on x-axis, 4 on y-axis, 4 off
        [[-3, 0], [-1, 0], [1, 0], [3, 0],
         [0, -3], [0, -1], [0, 1], [0, 3],
         [2, 2], [-2, 2], [2, -2], [-2, -2]],
        # 3x4 with shifted rows
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [1, 1], [2, 1], [3, 1], [4, 1],
         [2, 2], [3, 2], [4, 2], [5, 2]],
        # Two parallel lines with 6 each, offset
        [[i, 0] for i in range(6)] + [[i + 1, 2] for i in range(6)],
        # Three lines: 5, 4, 3 points
        [[i, 0] for i in range(5)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(3)],
        # 12 points forming a regular dodecagon (approximate)
        [[11, 0], [8, 8], [0, 11], [-8, 8], [-11, 0], [-8, -8],
         [0, -11], [8, -8], [5, 5], [-5, 5], [-5, -5], [5, -5]],
        # 6 points on two lines forming an X, plus 6 on two more
        [[-3, 0], [-2, 0], [-1, 0], [1, 0], [2, 0], [3, 0],
         [0, -3], [0, -2], [0, -1], [0, 1], [0, 2], [0, 3]],
    ]
    
    best_points = None
    best_score = float('inf')
    
    for s in structured_seeds:
        if time.time() > DEADLINE - 5:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]
    
    # Try random starting configurations
    for seed in range(300):
        if time.time() > DEADLINE - 5:
            break
        random.seed(seed + 42)
        points = []
        for _ in range(12):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            points.append([x, y])
        if not _is_valid(points):
            continue
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    # Local search with incremental updates
    points = [p[:] for p in best_points]
    score = best_score
    line_map = _build_line_map(points)
    
    step_size = 20
    no_improve = 0
    max_no_improve = 5000
    iteration = 0
    
    while iteration < 200000:
        if time.time() > DEADLINE - 3:
            break
        if score <= 6:  # theoretical minimum
            break
        if no_improve > max_no_improve:
            # Restart with a different configuration
            if best_score < score:
                points = [p[:] for p in best_points]
                score = best_score
                line_map = _build_line_map(points)
            else:
                # Try a new random start
                random.seed(iteration % 5000)
                points = []
                for _ in range(12):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    points.append([x, y])
                if not _is_valid(points):
                    points = [p[:] for p in best_points]
                    score = best_score
                    line_map = _build_line_map(points)
                else:
                    score = count_ordinary_lines(points)
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                    line_map = _build_line_map(points)
            no_improve = 0
            iteration += 1
            continue
        
        # Choose move type: 0 = single point move, 1 = two point swap
        move_type = random.random()
        
        if move_type < 0.65:
            # Single point move
            idx = random.randint(0, 11)
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                iteration += 1
                continue
            old_x, old_y = points[idx]
            new_x, new_y = old_x + dx, old_y + dy
            new_pt = [new_x, new_y]
            
            new_score, new_line_map = _fast_move(points, idx, new_pt, line_map)
            if new_score is None:
                no_improve += 1
                iteration += 1
                continue
            
            if new_score < score:
                points[idx] = new_pt
                score = new_score
                line_map = new_line_map
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            elif new_score == score:
                # Accept with small probability (simulated annealing-like)
                if random.random() < 0.1:
                    points[idx] = new_pt
                    line_map = new_line_map
                no_improve += 1
            else:
                no_improve += 1
        else:
            # Two point move
            i1, i2 = random.sample(range(12), 2)
            dx1 = random.randint(-step_size, step_size)
            dy1 = random.randint(-step_size, step_size)
            dx2 = random.randint(-step_size, step_size)
            dy2 = random.randint(-step_size, step_size)
            if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                iteration += 1
                continue
            old1 = points[i1][:]
            old2 = points[i2][:]
            new1 = [old1[0] + dx1, old1[1] + dy1]
            new2 = [old2[0] + dx2, old2[1] + dy2]
            
            new_score, new_line_map = _fast_two_point_move(points, i1, i2, new1, new2, line_map)
            if new_score is None:
                no_improve += 1
                iteration += 1
                continue
            
            if new_score < score:
                points[i1] = new1
                points[i2] = new2
                score = new_score
                line_map = new_line_map
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_points = [p[:] for p in points]
            elif new_score == score:
                if random.random() < 0.1:
                    points[i1] = new1
                    points[i2] = new2
                    line_map = new_line_map
                no_improve += 1
            else:
                no_improve += 1
        
        # Decrease step size periodically
        if iteration % 10000 == 0 and step_size > 1:
            step_size = max(1, step_size // 2)
        
        iteration += 1
    
    # Final polish pass with small steps and incremental updates
    points = [p[:] for p in best_points]
    score = best_score
    line_map = _build_line_map(points)
    step_size = 1
    polish_iter = 0
    
    while polish_iter < 30000:
        if time.time() > DEADLINE - 1:
            break
        if score <= 6:
            break
        
        idx = random.randint(0, 11)
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            polish_iter += 1
            continue
        old_x, old_y = points[idx]
        new_x, new_y = old_x + old_y * 0, old_y + dy  # fix: should be old_x + dx
        new_x, new_y = old_x + dx, old_y + dy
        new_pt = [new_x, new_y]
        
        new_score, new_line_map = _fast_move(points, idx, new_pt, line_map)
        if new_score is None:
            polish_iter += 1
            continue
        
        if new_score < score:
            points[idx] = new_pt
            score = new_score
            line_map = new_line_map
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
        else:
            pass  # revert (points unchanged)
        
        polish_iter += 1
    
    return best_points
