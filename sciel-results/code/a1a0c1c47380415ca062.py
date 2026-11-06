import time
import random
import math
from collections import defaultdict

def _score_points(points):
    """
    Returns (num_exact_4_lines, has_5_collinear, three_point_lines).
    points: list of 16 [x, y]
    """
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            # Ensure consistent direction (e.g., dx > 0 or dx == 0 and dy > 0)
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: A*x + B*y + C = 0
            # A = dy, B = -dx, C = -(A*x1 + B*y1)
            A = dy_n
            B = -dx_n
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C) by gcd
            g2 = math.gcd(abs(A), abs(B))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
                
            # Final sign normalization
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            lines[key].add(i)
            lines[key].add(j)
            
    exact_4 = 0
    has_5 = False
    three_point_lines = []
    
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
        elif cnt == 3:
            three_point_lines.append((key, list(pts)))
            
    return exact_4, has_5, three_point_lines

def _compute_lines(points):
    """Compute all lines and their point sets from 16 points."""
    n = len(points)
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            A = dy_n
            B = -dx_n
            C = -(A * x1 + B * y1)
            
            g2 = math.gcd(abs(A), abs(B))
            if g2 > 0:
                A //= g2
                B //= g2
                C //= g2
                
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            lines[key].add(i)
            lines[key].add(j)
            
    return lines

def _find_completion_candidates(points, lines):
    """
    Find lines with exactly 3 points and identify the 4th point position
    that would complete them to a 4-point line.
    Returns list of (line_key, missing_index, candidate_positions)
    """
    n = len(points)
    candidates = []
    
    for key, pts in lines.items():
        if len(pts) != 3:
            continue
        
        p1, p2, p3 = points[list(pts)[0]], points[list(pts)[1]], points[list(pts)[2]]
        
        # The line is determined by any two of these three points
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx_n = dx // g
        dy_n = dy // g
        
        # Direction vector for the line
        # A point (x, y) is on this line if (x-x1)*dy_n - (y-y1)*dx_n == 0
        
        # Find integer points on this line within bounds
        # Parametric: (x1 + t*dx_n, y1 + t*dy_n) for integer t
        # But we need to find ALL integer points, not just those at integer t from this parameterization
        
        # Actually, since (dx_n, dy_n) is the primitive direction, all integer points
        # on the line are of the form (x1 + k*dx_n, y1 + k*dy_n) for integer k
        
        # Find range of k such that |x1 + k*dx_n| <= 1000 and |y1 + k*dy_n| <= 1000
        k_min = float('inf')
        k_max = float('-inf')
        
        if dx_n != 0:
            k_min = max(k_min, math.ceil((-1000 - x1) / dx_n) if dx_n > 0 else math.floor((-1000 - x1) / dx_n))
            k_max = min(k_max, math.floor((1000 - x1) / dx_n) if dx_n > 0 else math.ceil((1000 - x1) / dx_n))
        if dy_n != 0:
            k_min = max(k_min, math.ceil((-1000 - y1) / dy_n) if dy_n > 0 else math.floor((-1000 - y1) / dy_n))
            k_max = min(k_max, math.floor((1000 - y1) / dy_n) if dy_n > 0 else math.ceil((1000 - y1) / dy_n))
        
        if k_min > k_max:
            continue
        
        # Generate all integer points on this line within bounds
        existing_points = set(tuple(p) for p in points)
        line_points_on_line = set()
        
        for k in range(int(k_min), int(k_max) + 1):
            x = x1 + k * dx_n
            y = y1 + k * dy_n
            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                line_points_on_line.add((x, y))
        
        # Filter to only points not already in our set
        new_candidates = line_points_on_line - existing_points
        
        if new_candidates:
            candidates.append((key, list(pts), new_candidates))
    
    return candidates

def _apply_move(current, move_type, rng):
    """
    Apply a move to current configuration.
    Returns (new_config, move_info) for potential rollback.
    """
    if move_type == 'completion':
        # Move a point to complete a 3-point line
        pass  # Handled in main loop
    
    elif move_type == 'single':
        idx = rng.randint(0, 15)
        old = current[idx][:]
        step = rng.randint(1, 20)
        dx = rng.randint(-step, step)
        dy = rng.randint(-step, step)
        if dx == 0 and dy == 0:
            dx = 1
        new_x = old[0] + dx
        new_y = old[1] + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            return current, None
            
        new_pt = [new_x, new_y]
        if any(current[k] == new_pt for k in range(16) if k != idx):
            return current, None
            
        current[idx] = new_pt
        return current, (idx, old)
        
    elif move_type == 'swap':
        i, j = rng.sample(range(16), 2)
        old_i = current[i][:]
        old_j = current[j][:]
        current[i], current[j] = current[j], current[i]
        return current, (i, j, old_i, old_j)
        
    else:
        return current, None

def construct():
    deadline = time.time() + 115
    
    # Initial configuration: 4x4 grid
    base = [[i * 10, j * 10] for i in range(4) for j in range(4)]
    
    best_points = [p[:] for p in base]
    best_score, _, _ = _score_points(base)
    
    rng = random.Random(42)
    
    T_start = 15.0
    T_end = 0.01
    
    for restart in range(80):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            current = [p[:] for p in base]
        else:
            # Start from best with perturbation
            current = [p[:] for p in best_points]
            for _ in range(rng.randint(5, 15)):
                idx = rng.randint(0, 15)
                current[idx][0] += rng.randint(-15, 15)
                current[idx][1] += rng.randint(-15, 15)
                current[idx][0] = max(-1000, min(1000, current[idx][0]))
                current[idx][1] = max(-1000, min(1000, current[idx][1]))
                # Ensure distinct
                for k in range(16):
                    if k != idx and current[k] == current[idx]:
                        current[idx][0] += rng.randint(1, 5)
                        break
        
        s, h5, _ = _score_points(current)
        if h5:
            # Try to fix
            for _ in range(100):
                lines = _compute_lines(current)
                found = None
                for key, pts in lines.items():
                    if len(pts) >= 5:
                        found = pts
                        break
                if not found:
                    break
                idx_to_move = rng.choice(list(found))
                old = current[idx_to_move][:]
                current[idx_to_move][0] += rng.randint(-3, 3)
                current[idx_to_move][1] += rng.randint(-3, 3)
                if abs(current[idx_to_move][0]) > 1000 or abs(current[idx_to_move][1]) > 1000:
                    current[idx_to_move] = old
                    continue
                if any(current[k] == current[idx_to_move] for k in range(16) if k != idx):
                    current[idx_to_move] = old
                    continue
                s, h5, _ = _score_points(current)
                if not h5:
                    break
            if h5:
                continue
        
        T = T_start
        current_score = s
        
        while time.time() < deadline - 2 and T > T_end:
            # Choose move type with bias toward completion moves
            r = rng.random()
            
            if r < 0.4:
                # Completion move: target a 3-point line
                lines = _compute_lines(current)
                candidates = _find_completion_candidates(current, lines)
                
                if candidates:
                    # Pick a random candidate line
                    key, pts_in_line, new_positions = rng.choice(candidates)
                    
                    # Pick a point to move (prefer one NOT on the target line)
                    points_on_line = set(pts_in_line)
                    points_off_line = [i for i in range(16) if i not in points_on_line]
                    
                    if points_off_line:
                        idx = rng.choice(points_off_line)
                    else:
                        # All points are on lines, pick random
                        idx = rng.randint(0, 15)
                    
                    old = current[idx][:]
                    
                    # Pick a target position from new_positions
                    target_x, target_y = rng.choice(list(new_positions))
                    
                    # Check if moving idx to target creates 5-collinear
                    current[idx] = [target_x, target_y]
                    
                    # Quick check for duplicates
                    if any(current[k] == [target_x, target_y] for k in range(16) if k != idx):
                        current[idx] = old
                        T *= 0.999
                        continue
                        
                    s_new, h5_new, _ = _score_points(current)
                    
                    if h5_new:
                        current[idx] = old
                        T *= 0.999
                        continue
                        
                    delta = s_new - current_score
                    if delta > 0:
                        current_score = s_new
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current]
                    else:
                        if rng.random() < math.exp(delta / T):
                            current_score = s_new
                        else:
                            current[idx] = old
                else:
                    # No completion candidates, fall back to single move
                    idx = rng.randint(0, 15)
                    old = current[idx][:]
                    step = rng.randint(1, 15)
                    dx = rng.randint(-step, step)
                    dy = rng.randint(-step, step)
                    if dx == 0 and dy == 0:
                        dx = 1
                    new_x = old[0] + dx
                    new_y = old[1] + dy
                    
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        T *= 0.999
                        continue
                        
                    new_pt = [new_x, new_y]
                    if any(current[k] == new_pt for k in range(16) if k != idx):
                        T *= 0.999
                        continue
                        
                    current[idx] = new_pt
                    s_new, h5_new, _ = _score_points(current)
                    
                    if h5_new:
                        current[idx] = old
                        T *= 0.999
                        continue
                        
                    delta = s_new - current_score
                    if delta > 0:
                        current_score = s_new
                        if current_score > best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current]
                    else:
                        if rng.random() < math.exp(delta / T):
                            current_score = s_new
                        else:
                            current[idx] = old
                    
            elif r < 0.7:
                # Single point move
                idx = rng.randint(0, 15)
                old = current[idx][:]
                step = rng.randint(1, 20)
                dx = rng.randint(-step, step)
                dy = rng.randint(-step, step)
                if dx == 0 and dy == 0:
                    dx = 1
                new_x = old[0] + dx
                new_y = old[1] + dy
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    T *= 0.999
                    continue
                    
                new_pt = [new_x, new_y]
                if any(current[k] == new_pt for k in range(16) if k != idx):
                    T *= 0.999
                    continue
                    
                current[idx] = new_pt
                s_new, h5_new, _ = _score_points(current)
                
                if h5_new:
                    current[idx] = old
                    T *= 0.999
                    continue
                    
                delta = s_new - current_score
                if delta > 0:
                    current_score = s_new
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current]
                else:
                    if rng.random() < math.exp(delta / T):
                        current_score = s_new
                    else:
                        current[idx] = old
                        
            else:
                # Swap two points
                i, j = rng.sample(range(16), 2)
                old_i = current[i][:]
                old_j = current[j][:]
                current[i], current[j] = current[j], current[i]
                
                s_new, h5_new, _ = _score_points(current)
                
                if h5_new:
                    current[i], current[j] = old_i, old_j
                    T *= 0.999
                    continue
                    
                delta = s_new - current_score
                if delta > 0:
                    current_score = s_new
                    if current_score > best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current]
                else:
                    if rng.random() < math.exp(delta / T):
                        current_score = s_new
                    else:
                        current[i], current[j] = old_i, old_j
            
            T *= 0.999

    return best_points
