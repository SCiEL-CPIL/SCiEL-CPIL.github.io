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
    Returns (score, valid).
    Score: number of lines with exactly 4 points.
    Valid: True if no 5 collinear, distinct, and within bounds.
    """
    n = len(points)
    if n != 16:
        return -1, False
    
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False
        pt = (x, y)
        if pt in seen:
            return -1, False
        seen.add(pt)
        
    line_map = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                return -1, False
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    exact_4_count = 0
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) > 4:
            return -1, False
        if len(pts_in_line) == 4:
            exact_4_count += 1
            
    return exact_4_count, True

def _grid_start():
    """4x4 grid centered at origin."""
    pts = []
    for x in range(-1, 3):
        for y in range(-1, 3):
            pts.append([x, y])
    return pts

def _random_start():
    """Random points within bounds."""
    return [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]

def _sheared_grid_start(seed):
    """
    Generates a sheared/scaled grid to provide a better starting point
    than a pure rectangular grid.
    """
    random.seed(seed)
    base = _grid_start()
    # Apply a random affine transformation
    # x' = ax + by + c
    # y' = dx + ey + f
    # Keep it simple: shear and scale
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

def _simulate_annealing(start_points, time_limit, seed=None):
    """
    Runs Simulated Annealing with structure-aware moves.
    """
    if seed is not None:
        random.seed(seed)
        
    current = [list(p) for p in start_points]
    current_score, valid = _evaluate(current)
    
    if not valid:
        # Try to fix initial state
        for _ in range(100):
            idx = random.randint(0, 15)
            current[idx] = [current[idx][0] + random.randint(-2, 2), 
                            current[idx][1] + random.randint(-2, 2)]
            current_score, valid = _evaluate(current)
            if valid:
                break
        if not valid:
            current = _grid_start()
            current_score, valid = _evaluate(current)
            if not valid:
                return current, 0

    best = [list(p) for p in current]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99995
    
    start_time = time.time()
    
    while time.time() < time_limit:
        # Generate move
        move_type = random.random()
        
        if move_type < 0.5:
            # Move Type 1: Pull to Line
            # Pick a point to move (P)
            idx = random.randint(0, 15)
            P = current[idx]
            
            # Pick two other points to define a line (Q, R)
            others = [i for i in range(16) if i != idx]
            q_idx, r_idx = random.sample(others, 2)
            Q = current[q_idx]
            R = current[r_idx]
            
            key = _line_key(Q, R)
            if key is None:
                continue
                
            A, B, C = key
            # Line: Ax + By = C
            # We want to move P to a point on this line.
            # Find the closest integer point on the line to P that is not occupied.
            # Parametric form:
            # If A != 0: x = (C - By) / A. 
            # This is complex. Instead, just try a few candidate points near P on the line.
            
            # Simple heuristic: Try to move P to align with Q and R.
            # Calculate the "target" position if we force P to be on the line QR.
            # This is a projection.
            
            # Alternative: Just try to move P to a random point on the line QR 
            # within a small radius of its current position.
            
            # Let's try a simpler "Align" move:
            # If Q and R are collinear with some existing point S, 
            # try to move P to S's position? No, distinctness.
            
            # Let's use a "Slide" move on an existing line.
            # If P is already on a line with at least 2 other points, slide it.
            
            # For now, let's do a "Targeted Jitter"
            # Move P by a small vector, but bias towards Q or R?
            # No, let's stick to the "Pull" concept:
            # Find the nearest integer point to P that lies on the line defined by Q and R.
            
            # Line: Ax + By = C
            # Minimize (x-Px)^2 + (y-Py)^2 subject to Ax+By=C and x,y integers.
            
            # This is a quadratic optimization.
            # Approximation: 
            # If A == 0: y = C/B. x can be anything.
            #   Best x is Px. y = C/B.
            #   If C/B is not integer, no solution.
            # If B == 0: x = C/A. y can be anything.
            #   Best y is Py. x = C/A.
            # If A, B != 0:
            #   We can iterate x near Px and check if y = (C - Ax)/B is integer.
            
            found_target = False
            for dx in range(-10, 11):
                nx = P[0] + dx
                if abs(nx) > 1000: continue
                # Check if there is an integer y such that A*nx + B*y = C
                # B*y = C - A*nx
                rem = C - A * nx
                if B != 0:
                    if rem % B == 0:
                        ny = rem // B
                        if abs(ny) <= 1000:
                            # Check if (nx, ny) is not occupied
                            if (nx, ny) not in [(p[0], p[1]) for p in current]:
                                trial = [list(p) for p in current]
                                trial[idx] = [nx, ny]
                                found_target = True
                                break
                else:
                    # B == 0, A != 0. x is fixed by line.
                    # We already checked nx. If A*nx == C, then any y works?
                    # No, if B=0, line is vertical x=C/A.
                    # We need nx == C/A.
                    if A * nx == C:
                        ny = P[1] # Keep y same? Or randomize?
                        # Actually if B=0, the line is x=const.
                        # We can choose any y. Let's keep y close to Py.
                        if (nx, ny) not in [(p[0], p[1]) for p in current]:
                            trial = [list(p) for p in current]
                            trial[idx] = [nx, ny]
                            found_target = True
                            break
            
            if not found_target:
                # Fallback to small jitter
                delta_x = random.randint(-3, 3)
                delta_y = random.randint(-3, 3)
                if delta_x == 0 and delta_y == 0:
                    continue
                new_pt = [P[0] + delta_x, P[1] + delta_y]
                if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                    continue
                trial = [list(p) for p in current]
                trial[idx] = new_pt
                
        elif move_type < 0.8:
            # Move Type 2: Small Jitter
            idx = random.randint(0, 15)
            delta_x = random.randint(-3, 3)
            delta_y = random.randint(-3, 3)
            if delta_x == 0 and delta_y == 0:
                continue
            new_pt = [current[idx][0] + delta_x, current[idx][1] + delta_y]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            trial = [list(p) for p in current]
            trial[idx] = new_pt
        else:
            # Move Type 3: Large Jump / Reset
            idx = random.randint(0, 15)
            new_pt = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            trial = [list(p) for p in current]
            trial[idx] = new_pt

        trial_score, trial_valid = _evaluate(trial)
        
        if trial_valid:
            delta = trial_score - current_score
            if delta > 0:
                current = trial
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best = [list(p) for p in current]
            else:
                if random.random() < math.exp(delta / T):
                    current = trial
                    current_score = trial_score
        else:
            # Reject invalid
            pass
        
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
            
        s, v = _evaluate(start)
        if not v:
             for _ in range(20):
                 idx = random.randint(0, 15)
                 start[idx] = [start[idx][0] + random.randint(-5, 5), 
                               start[idx][1] + random.randint(-5, 5)]
                 s, v = _evaluate(start)
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
        best_score, _ = _evaluate(best_points)
        
    return [list(p) for p in best_points]
