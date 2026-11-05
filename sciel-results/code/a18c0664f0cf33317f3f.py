import math
import time
import random
from math import gcd
from collections import defaultdict

def _evaluate(points):
    """Evaluate a set of 16 points. Returns (is_valid, score)."""
    n = len(points)
    if n != 16:
        return False, 0
    
    point_set = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
    
    # Count points on each line using pair enumeration
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            ndx = dx // g
            ndy = dy // g
            if ndx < 0 or (ndx == 0 and ndy < 0):
                ndx = -ndx
                ndy = -ndy
            c = ndy * x1 - ndx * y1
            key = (ndx, ndy, c)
            line_counts[key] += 1
    
    score = 0
    for count in line_counts.values():
        if count >= 10:  # C(5,2) = 10 means 5+ collinear
            return False, 0
        if count == 6:   # C(4,2) = 6 means exactly 4 collinear
            score += 1
    
    return True, score

def _evaluate_delta(points, moved_indices, old_positions):
    """Fast evaluation: only recompute lines involving moved points.
    Returns (is_valid, score) or None if we can't determine validity."""
    n = len(points)
    
    # Check basic validity
    point_set = set()
    for i, p in enumerate(points):
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        pt = (x, y)
        if pt in point_set:
            return False, 0
        point_set.add(pt)
    
    # Build full line count - for 16 points, full re-eval is fast enough
    # This is a fallback; in practice we'll use full eval
    return _evaluate(points)

def _generate_grid_config():
    """Generate 4x4 grid centered at origin."""
    return [[x, y] for x in range(-1, 3) for y in range(-1, 3)]

def _try_shift_point(points, idx, dx, dy):
    """Try shifting point idx by (dx, dy). Returns new points or None if invalid."""
    new_points = [p[:] for p in points]
    nx = points[idx][0] + dx
    ny = points[idx][1] + dy
    if abs(nx) > 1000 or abs(ny) > 1000:
        return None
    new_points[idx] = [nx, ny]
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with 4x4 grid (score 10)
    current_points = _generate_grid_config()
    is_valid, current_score = _evaluate(current_points)
    if not is_valid:
        current_points = [[x, y] for x in range(4) for y in range(4)]
        is_valid, current_score = _evaluate(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Simulated annealing with proper state tracking
    T = 50.0
    cooling_rate = 0.99995
    min_T = 0.5
    
    steps_since_improve = 0
    max_steps_no_improve = 3000
    
    while time.time() < deadline:
        prev_points = [p[:] for p in current_points]
        prev_score = current_score
        
        # Generate a move
        r = random.random()
        
        if r < 0.6:
            # Small single-point move
            idx = random.randint(0, 15)
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            new_points = _try_shift_point(current_points, idx, dx, dy)
            if new_points is None:
                continue
            current_points = new_points
        elif r < 0.8:
            # Medium single-point move
            idx = random.randint(0, 15)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            new_points = _try_shift_point(current_points, idx, dx, dy)
            if new_points is None:
                continue
            current_points = new_points
        elif r < 0.9:
            # Two-point coordinated move
            i, j = random.sample(range(16), 2)
            di = random.randint(-8, 8)
            dj = random.randint(-8, 8)
            new_points = [p[:] for p in current_points]
            ni_x = current_points[i][0] + di
            ni_y = current_points[i][1] + dj
            nj_x = current_points[j][0] + random.randint(-8, 8)
            nj_y = current_points[j][1] + random.randint(-8, 8)
            if abs(ni_x) <= 1000 and abs(ni_y) <= 1000 and abs(nj_x) <= 1000 and abs(nj_y) <= 1000:
                new_points[i] = [ni_x, ni_y]
                new_points[j] = [nj_x, nj_y]
                current_points = new_points
            else:
                continue
        else:
            # Large jump: reset one point to random position
            idx = random.randint(0, 15)
            nx = random.randint(-50, 50)
            ny = random.randint(-50, 50)
            new_points = [p[:] for p in current_points]
            new_points[idx] = [nx, ny]
            current_points = new_points
        
        is_valid, new_score = _evaluate(current_points)
        
        if is_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                steps_since_improve = 0
                current_score = new_score
            elif new_score == current_score:
                # Accept equal moves (helps exploration)
                current_score = new_score
                steps_since_improve += 1
            else:
                # Worse move: accept with probability exp(-delta/T)
                delta = current_score - new_score
                if delta <= 0:
                    current_score = new_score
                else:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        current_points = prev_points
                        current_score = prev_score
        else:
            # Invalid: revert
            current_points = prev_points
            current_score = prev_score
        
        # Cooling
        T = max(min_T, T * cooling_rate)
        
        # Restart if stuck
        steps_since_improve += 1
        if steps_since_improve > max_steps_no_improve:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 30.0
            steps_since_improve = 0
    
    return best_points
