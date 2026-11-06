"""
Optimized solver for the 16-point integer collinearity problem.
Uses Simulated Annealing with structure-aware moves (Pull-to-Line)
to maximize the number of lines with exactly 4 points.
"""
import time
import random
import math
from math import gcd

_DEADLINE = 110.0  # Safety margin before 120s limit

def _line_key(p1, p2):
    """
    Returns a canonical tuple (A, B, C) representing the line Ax + By = C.
    Normalization:
    1. A, B are integers.
    2. gcd(A, B) = 1.
    3. A > 0, or (A == 0 and B > 0).
    """
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None

    A = dy
    B = -dx
    C = A * x1 + B * y1
    
    g = gcd(abs(A), abs(B))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _evaluate(points):
    """
    Evaluates the configuration.
    Returns (score, valid, line_map).
    Score: number of lines with exactly 4 points.
    Valid: True if no 5 collinear, distinct, and within bounds.
    line_map: dict of line_key -> set of point indices.
    """
    n = len(points)
    if n != 16:
        return -1, False, {}
    
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False, {}
        pt = (x, y)
        if pt in seen:
            return -1, False, {}
        seen.add(pt)
        
    line_map = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                return -1, False, {}
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    exact_4_count = 0
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) > 4:
            return -1, False, line_map
        if len(pts_in_line) == 4:
            exact_4_count += 1
            
    return exact_4_count, True, line_map

def _get_line_candidates(line_key, occupied_set, bounds=1000, max_search_range=50):
    """
    Finds integer points on the line Ax+By=C that are not occupied.
    Returns a list of candidate points.
    """
    A, B, C = line_key
    candidates = []
    
    if A == 0 and B == 0:
        return []
        
    if A == 0:
        # By = C -> y = C/B
        if C % B != 0:
            return []
        y_val = C // B
        if abs(y_val) > bounds:
            return []
        # x can be anything. We only care about points near the current configuration?
        # Actually, we want to move a point P to a candidate.
        # We don't know P yet. This function is generic.
        # But we can't return all x in [-1000, 1000].
        # We need to pass a reference point or just return a few.
        # For now, this helper is incomplete for the move logic.
        # We will implement the move logic directly in the SA loop.
        pass
        
    return candidates

def _grid_start():
    """4x4 grid centered at origin."""
    pts = []
    for x in range(-1, 3):
        for y in range(-1, 3):
            pts.append([x, y])
    return pts

def _sheared_grid_start(seed):
    """
    Generates a sheared/scaled grid to provide a better starting point.
    """
    random.seed(seed)
    base = _grid_start()
    s = random.randint(1, 3)
    scale = random.randint(2, 5)
    offset_x = random.randint(-50, 50)
    offset_y = random.randint(-50, 50)
    
    pts = []
    for x, y in base:
        nx = (x * scale) + (y * s) + offset_x
        ny = (y * scale) + offset_y
        pts.append([nx, ny])
    return pts

def _random_start():
    """Random points within bounds."""
    return [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]

def _simulate_annealing(start_points, time_limit, seed=None):
    """
    Runs Simulated Annealing with structure-aware moves.
    """
    if seed is not None:
        random.seed(seed)
        
    current = [list(p) for p in start_points]
    current_score, valid, current_line_map = _evaluate(current)
    
    if not valid:
        # Try to fix initial state
        for _ in range(100):
            idx = random.randint(0, 15)
            current[idx] = [current[idx][0] + random.randint(-2, 2), 
                            current[idx][1] + random.randint(-2, 2)]
            current_score, valid, current_line_map = _evaluate(current)
            if valid:
                break
        if not valid:
            current = _grid_start()
            current_score, valid, current_line_map = _evaluate(current)
            if not valid:
                return current, 0

    best = [list(p) for p in current]
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.99995
    
    start_time = time.time()
    
    # Precompute occupied set for fast lookup
    occupied = set((p[0], p[1]) for p in current)
    
    while time.time() < time_limit:
        move_type = random.random()
        
        if move_type < 0.6:
            # Move Type 1: Pull to Line (Complete a 3-point line)
            # Find a 3-point line in current_line_map
            three_point_lines = [key for key, pts in current_line_map.items() if len(pts) == 3]
            
            if three_point_lines:
                key = random.choice(three_point_lines)
                A, B, C = key
                line_pts_idx = current_line_map[key]
                
                # Pick a point P to move (not on this line)
                others = [i for i in range(16) if i not in line_pts_idx]
                if others:
                    idx = random.choice(others)
                    P = current[idx]
                    
                    # Find candidate positions on the line Ax+By=C
                    # We want to move P to a point D on the line.
                    # D should be "close" to P or just any valid point.
                    # Let's try to find integer solutions (x,y) to Ax+By=C
                    # that are not in occupied.
                    
                    found_target = False
                    # Search range around P
                    # If A != 0, x = (C - By)/A. 
                    # If B != 0, y = (C - Ax)/B.
                    
                    # Iterate x in a range around P[0]
                    # We need to be careful with the range.
                    # Since we are moving P, D can be anywhere, but closer is better for "local" moves.
                    # However, to create new lines, we might need to jump.
                    # Let's try a few candidates.
                    
                    # If A == 0: y is fixed.
                    if A == 0:
                        if C % B != 0:
                            continue
                        y_val = C // B
                        if abs(y_val) > 1000:
                            continue
                        # x can be anything. Let's pick a few x values.
                        # Maybe x close to P[0]?
                        for dx in [-5, -2, 0, 2, 5, 10, -10]:
                            x_val = P[0] + dx
                            if abs(x_val) > 1000: continue
                            if (x_val, y_val) in occupied:
                                continue
                            # Trial move
                            trial = [list(p) for p in current]
                            trial[idx] = [x_val, y_val]
                            
                            # Update occupied
                            occupied.remove(tuple(P))
                            occupied.add((x_val, y_val))
                            
                            score, valid, lm = _evaluate(trial)
                            
                            occupied.remove((x_val, y_val))
                            occupied.add(tuple(P))
                            
                            if valid:
                                found_target = True
                                delta = score - current_score
                                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                    current = trial
                                    current_score = score
                                    current_line_map = lm
                                    if score > best_score:
                                        best_score = score
                                        best = [list(p) for p in current]
                                    break
                    else:
                        # A != 0.
                        # x = (C - By) / A.
                        # We can iterate y in a range around P[1] or x in a range around P[0].
                        # Let's iterate x.
                        # y = (C - Ax) / B.
                        # If B == 0, x is fixed.
                        if B == 0:
                            if C % A != 0:
                                continue
                            x_val = C // A
                            if abs(x_val) > 1000:
                                continue
                            for dy in [-5, -2, 0, 2, 5, 10, -10]:
                                y_val = P[1] + dy
                                if abs(y_val) > 1000: continue
                                if (x_val, y_val) in occupied:
                                    continue
                                trial = [list(p) for p in current]
                                trial[idx] = [x_val, y_val]
                                
                                occupied.remove(tuple(P))
                                occupied.add((x_val, y_val))
                                
                                score, valid, lm = _evaluate(trial)
                                
                                occupied.remove((x_val, y_val))
                                occupied.add(tuple(P))
                                
                                if valid:
                                    found_target = True
                                    delta = score - current_score
                                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                        current = trial
                                        current_score = score
                                        current_line_map = lm
                                        if score > best_score:
                                            best_score = score
                                            best = [list(p) for p in current]
                                        break
                        else:
                            # Both A and B non-zero.
                            # Iterate x around P[0]
                            for dx in [-10, -5, -2, -1, 1, 2, 5, 10]:
                                x_val = P[0] + dx
                                if abs(x_val) > 1000: continue
                                rem = C - A * x_val
                                if rem % B != 0:
                                    continue
                                y_val = rem // B
                                if abs(y_val) > 1000:
                                    continue
                                if (x_val, y_val) in occupied:
                                    continue
                                
                                trial = [list(p) for p in current]
                                trial[idx] = [x_val, y_val]
                                
                                occupied.remove(tuple(P))
                                occupied.add((x_val, y_val))
                                
                                score, valid, lm = _evaluate(trial)
                                
                                occupied.remove((x_val, y_val))
                                occupied.add(tuple(P))
                                
                                if valid:
                                    found_target = True
                                    delta = score - current_score
                                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                        current = trial
                                        current_score = score
                                        current_line_map = lm
                                        if score > best_score:
                                            best_score = score
                                            best = [list(p) for p in current]
                                        break
                                 
        elif move_type < 0.9:
            # Move Type 2: Small Jitter
            idx = random.randint(0, 15)
            delta_x = random.randint(-3, 3)
            delta_y = random.randint(-3, 3)
            if delta_x == 0 and delta_y == 0:
                continue
            new_pt = [current[idx][0] + delta_x, current[idx][1] + delta_y]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            if (new_pt[0], new_pt[1]) in occupied:
                continue
            
            trial = [list(p) for p in current]
            trial[idx] = new_pt
            
            occupied.remove(tuple(current[idx]))
            occupied.add(tuple(new_pt))
            
            trial_score, trial_valid, trial_lm = _evaluate(trial)
            
            occupied.remove(tuple(new_pt))
            occupied.add(tuple(current[idx]))
            
            if trial_valid:
                delta = trial_score - current_score
                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                    current = trial
                    current_score = trial_score
                    current_line_map = trial_lm
                    if current_score > best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
        else:
            # Move Type 3: Large Jump
            idx = random.randint(0, 15)
            new_pt = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            if (new_pt[0], new_pt[1]) in occupied:
                continue
                
            trial = [list(p) for p in current]
            trial[idx] = new_pt
            
            occupied.remove(tuple(current[idx]))
            occupied.add(tuple(new_pt))
            
            trial_score, trial_valid, trial_lm = _evaluate(trial)
            
            occupied.remove(tuple(new_pt))
            occupied.add(tuple(current[idx]))
            
            if trial_valid:
                delta = trial_score - current_score
                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                    current = trial
                    current_score = trial_score
                    current_line_map = trial_lm
                    if current_score > best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
        
        T = max(T * cooling, T_min)
        
    return best, best_score

def construct():
    start_time = time.time()
    remaining = _DEADLINE - start_time
    
    best_points = None
    best_score = -1
    
    # Define multiple starting configurations
    starts = [
        _grid_start(),
        _sheared_grid_start(1),
        _sheared_grid_start(2),
        _sheared_grid_start(3),
        _random_start(),
    ]
    
    num_starts = len(starts)
    time_per_start = remaining / (num_starts + 1)
    
    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 5:
            break
            
        s, v, _ = _evaluate(start)
        if not v:
             for _ in range(20):
                 idx = random.randint(0, 15)
                 start[idx] = [start[idx][0] + random.randint(-5, 5), 
                               start[idx][1] + random.randint(-5, 5)]
                 s, v, _ = _evaluate(start)
                 if v: break
             if not v:
                 continue

        local_best, local_score = _simulate_annealing(start, time.time() + time_per_start, seed=100+i)
        
        if local_score > best_score:
            best_score = local_score
            best_points = local_best
            
    if best_points is not None and time.time() < _DEADLINE - 2:
        time_left = _DEADLINE - 2 - time.time()
        if time_left > 5:
            final_best, final_score = _simulate_annealing(best_points, time.time() + time_left, seed=999)
            if final_score > best_score:
                best_score = final_score
                best_points = final_best
    else:
        best_points = _grid_start()
        best_score, _, _ = _evaluate(best_points)
        
    return [list(p) for p in best_points]
