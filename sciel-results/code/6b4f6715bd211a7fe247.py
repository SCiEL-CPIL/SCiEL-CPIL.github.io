import random
import time
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    
    g = gcd(abs(A), abs(B))
    if g == 0:
        g = abs(C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _build_line_index(points):
    """Build mapping from line_key -> set of point indices on that line."""
    n = len(points)
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _get_line_key(points[i], points[j])
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return line_points

def _evaluate_from_index(points, line_points):
    """Count lines with exactly 4 points, check for 5+. Returns (valid, score)."""
    valid = True
    score = 0
    for key, idx_set in line_points.items():
        cnt = len(idx_set)
        if cnt >= 5:
            valid = False
            break
        elif cnt == 4:
            score += 1
    return valid, score

def _incremental_update(points, line_points, old_idx, old_pt, new_pt):
    """
    Update line_points after moving point old_idx from old_pt to new_pt.
    Mutates line_points in place.
    """
    n = len(points)
    # Remove old point from all lines it was on
    lines_to_remove = []
    for key, idx_set in line_points.items():
        if old_idx in idx_set:
            lines_to_remove.append(key)
    
    for key in lines_to_remove:
        idx_set = line_points[key]
        idx_set.discard(old_idx)
        if len(idx_set) <= 1:
            del line_points[key]
    
    # Add new point to lines it's on
    for j in range(n):
        if j == old_idx:
            continue
        key = _get_line_key(new_pt, points[j])
        if key not in line_points:
            line_points[key] = set()
        line_points[key].add(old_idx)
        line_points[key].add(j)
    
    return line_points

def _try_move(points, line_points, idx, new_pt, current_score):
    """
    Try moving point idx to new_pt. Returns (accepted, new_line_points, new_score, is_valid).
    """
    old_pt = points[idx][:]
    n = len(points)
    
    # Check distinctness
    for i, p in enumerate(points):
        if i != idx:
            if p[0] == new_pt[0] and p[1] == new_pt[1]:
                return False, line_points, current_score, True
    
    # Apply move tentatively
    points[idx] = new_pt
    
    # Incremental update
    new_line_points = _incremental_update(points, line_points, idx, old_pt, new_pt)
    
    # Evaluate
    is_valid, new_score = _evaluate_from_index(points, new_line_points)
    
    return (is_valid and new_score > current_score), new_line_points, new_score, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s kill
    
    # Initial: 4x4 grid - has the affine plane of order 4 structure
    # 20 lines of exactly 4 points each
    points = [[x, y] for x in range(4) for y in range(4)]
    
    line_points = _build_line_index(points)
    is_valid, current_score = _evaluate_from_index(points, line_points)
    
    best_points = [p[:] for p in points]
    best_score = current_score
    
    if not is_valid:
        return best_points
    
    random.seed(42)
    
    # Simulated annealing inspired local search
    # Temperature schedule: start high, decay over time
    T_start = 1.0
    T_min = 0.01
    total_time = 110.0
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        # Linear cooling
        T = max(T_min, T_start * (1.0 - elapsed / total_time))
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Generate candidate moves with varying scales
        mode = random.random()
        if mode < 0.5:
            # Small perturbation - stay close to preserve structure
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        elif mode < 0.8:
            # Medium jump
            new_x = random.randint(-6, 6)
            new_y = random.randint(-6, 6)
        elif mode < 0.95:
            # Larger jump for exploration
            new_x = random.randint(-15, 15)
            new_y = random.randint(-15, 15)
        else:
            # Big jump for restart-like exploration
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Try the move
        accepted, new_line_points, new_score, is_valid = _try_move(
            points, line_points, idx, new_point, current_score
        )
        
        # Acceptance: always accept improvements, sometimes accept plateaus/worsening
        if is_valid and new_score > current_score:
            accept = True
        elif is_valid and new_score == current_score and random.random() < 0.2:
            accept = True
        elif is_valid and new_score < current_score and random.random() < T * 0.1:
            # Occasionally accept worsening to escape local optima
            accept = True
        else:
            accept = False
        
        if accept:
            current_score = new_score
            line_points = new_line_points
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            # Rebuild line_points from scratch to ensure consistency
            line_points = _build_line_index(points)
            # Re-evaluate to sync state
            is_valid, current_score = _evaluate_from_index(points, line_points)
    
    return best_points
