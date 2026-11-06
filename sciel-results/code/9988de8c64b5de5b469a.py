"""Structural change: Line-completion search with simulated annealing."""
import time
import random
import math
from fractions import Fraction

_START_TIME = None
_TIME_LIMIT = 140.0  # safety margin for 150s limit

def _time_left():
    if _START_TIME is None:
        return _TIME_LIMIT
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _score(points):
    """
    Returns (num_exact4_lines, valid).
    Efficient line counting using normalized line keys.
    """
    n = 16
    line_masks = {}
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            xj = xs[j]
            yj = ys[j]
            dx = xj - xi
            dy = yj - yi
            
            g = math.gcd(dx, dy)
            if g < 0:
                g = -g
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            a = dy_n
            b = -dx_n
            c = a * xi + b * yi
            
            g2 = math.gcd(abs(a), abs(b))
            if g2:
                a //= g2
                b //= g2
                c //= g2
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            
            mask = (1 << i) | (1 << j)
            if key in line_masks:
                line_masks[key] |= mask
            else:
                line_masks[key] = mask
                
    exact4 = 0
    for mask in line_masks.values():
        cnt = mask.bit_count()
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            return 0, False
            
    return exact4, True

def _inbounds(p):
    return -1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000

def _valid_config(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if not _inbounds(p):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    _, ok = _score(points)
    return ok

def _get_line_info(points):
    """
    Returns dict: line_key -> (mask, count)
    Also returns list of (line_key, mask, count) for lines with 3 points.
    """
    n = 16
    line_masks = {}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            xj = xs[j]
            yj = ys[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(dx, dy)
            if g < 0:
                g = -g
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            a = dy_n
            b = -dx_n
            c = a * xi + b * yi
            g2 = math.gcd(abs(a), abs(b))
            if g2:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            mask = (1 << i) | (1 << j)
            if key in line_masks:
                line_masks[key] |= mask
            else:
                line_masks[key] = mask
    
    three_lines = []
    for key, mask in line_masks.items():
        cnt = mask.bit_count()
        if cnt == 3:
            three_lines.append((key, mask, cnt))
    
    return line_masks, three_lines

def _find_completion_point(points, line_key, mask):
    """
    Given a line (a,b,c) and 3 points on it (mask), find the 4th integer point
    on this line that is close to the 3 existing points.
    Returns candidate point or None.
    """
    a, b, c = line_key
    # Line: a*x + b*y = c
    # Find integer points on this line near the 3 existing points
    
    # Get the 3 existing points
    pts = []
    for i in range(16):
        if mask & (1 << i):
            pts.append((points[i][0], points[i][1]))
    
    if len(pts) != 3:
        return None
    
    # Compute centroid
    cx = sum(p[0] for p in pts) / 3.0
    cy = sum(p[1] for p in pts) / 3.0
    
    # Find integer points on the line near centroid
    # a*x + b*y = c
    candidates = []
    
    if a == 0:
        # b*y = c -> y = c/b
        if c % b == 0:
            y = c // b
            # x can be anything, but we want near cx
            for x in range(int(cx) - 5, int(cx) + 6):
                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                    candidates.append((x, y))
    elif b == 0:
        # a*x = c -> x = c/a
        if c % a == 0:
            x = c // a
            for y in range(int(cy) - 5, int(cy) + 6):
                if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                    candidates.append((x, y))
    else:
        # General case: find integer solutions to a*x + b*y = c
        # Use extended GCD
        g = math.gcd(a, b)
        if c % g != 0:
            return None
        
        a1 = a // g
        b1 = b // g
        c1 = c // g
        
        # Find one solution using extended Euclidean
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
        
        g_val, x0, y0 = egcd(a1, b1)
        # x0*a1 + y0*b1 = 1
        # So x0*c1*a1 + y0*c1*b1 = c1
        x_part = x0 * c1
        y_part = y0 * c1
        
        # General solution: x = x_part + b1*t, y = y_part - a1*t
        # Find t such that (x, y) is near (cx, cy)
        best_t = None
        best_dist = float('inf')
        for t in range(-10, 11):
            x = x_part + b1 * t
            y = y_part - a1 * t
            if -1000 <= x <= 1000 and -1000 <= y <= 1000:
                dist = (x - cx)**2 + (y - cy)**2
                if dist < best_dist:
                    best_dist = dist
                    best_t = t
        
        if best_t is not None:
            x = x_part + b1 * best_t
            y = y_part - a1 * best_t
            candidates.append((x, y))
    
    # Filter: not already occupied, and not one of the 3 points
    occupied = set()
    for i in range(16):
        occupied.add((points[i][0], points[i][1]))
    
    valid_candidates = []
    for (x, y) in candidates:
        if (x, y) not in occupied:
            valid_candidates.append((x, y))
    
    return valid_candidates[0] if valid_candidates else None

def _line_completion_move(points, rng):
    """
    Try to complete a 3-point line to a 4-point line by moving a point.
    Returns new configuration or None.
    """
    line_masks, three_lines = _get_line_info(points)
    
    if not three_lines:
        return None
    
    # Pick a random 3-line
    line_key, mask, cnt = rng.choice(three_lines)
    
    # Find completion point
    comp = _find_completion_point(points, line_key, mask)
    if comp is None:
        return None
    
    # Find a point to move: prefer a point NOT on this line
    # and not on any other 4-line (to minimize damage)
    # Count how many 4-lines each point is on
    point_4line_count = [0] * 16
    for lkey, lmask in line_masks.items():
        if lmask.bit_count() == 4:
            for i in range(16):
                if lmask & (1 << i):
                    point_4line_count[i] += 1
    
    # Points on the 3-line
    on_line = [i for i in range(16) if mask & (1 << i)]
    # Points not on the 3-line, sorted by 4-line count (ascending)
    not_on_line = [i for i in range(16) if i not in on_line]
    not_on_line.sort(key=lambda i: point_4line_count[i])
    
    # Try moving the point with fewest 4-lines
    for idx in not_on_line:
        cand = [list(p) for p in points]
        cand[idx] = list(comp)
        
        # Check validity
        seen = set()
        valid = True
        for p in cand:
            if not _inbounds(p):
                valid = False
                break
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        
        if not valid:
            continue
        
        s, ok = _score(cand)
        if ok:
            return cand
    
    return None

def _perturb_point(points, rng):
    """
    Small random perturbation of one point.
    """
    cand = [list(p) for p in points]
    i = rng.randint(0, 15)
    dx = rng.randint(-8, 8)
    dy = rng.randint(-8, 8)
    new_p = [cand[i][0] + dx, cand[i][1] + dy]
    
    if not _inbounds(new_p):
        return None
    if any((new_p[0] == p[0] and new_p[1] == p[1]) and idx != i 
           for idx, p in enumerate(cand)):
        return None
    cand[i] = new_p
    return cand

def _simulated_annealing(init_pts, rng, max_time):
    """
    Simulated annealing with line-completion and perturbation moves.
    """
    cur = [list(p) for p in init_pts]
    cur_score, valid = _score(cur)
    if not valid:
        return None, -1
    
    best = [list(p) for p in cur]
    best_score = cur_score
    
    start = time.time()
    T = 5.0  # initial temperature
    cooling = 0.999
    
    while time.time() - start < max_time:
        if _time_left() < 2.0:
            break
        
        move_type = rng.random()
        
        if move_type < 0.4:
            cand = _line_completion_move(cur, rng)
        else:
            cand = _perturb_point(cur, rng)
        
        if cand is None:
            continue
        
        cs, ok = _score(cand)
        if not ok:
            continue
        
        delta = cs - cur_score
        
        if delta >= 0:
            cur = cand
            cur_score = cs
            if cur_score > best_score:
                best_score = cur_score
                best = [list(p) for p in cur]
        else:
            # Accept with probability exp(delta / T)
            if rng.random() < math.exp(delta / T):
                cur = cand
                cur_score = cs
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best, best_score

def _grid_init(scale=1):
    """4x4 grid scaled."""
    return [[x * scale, y * scale] for x in range(4) for y in range(4)]

def _random_init(rng, bound=300):
    pts = []
    seen = set()
    while len(pts) < 16:
        x = rng.randint(-bound, bound)
        y = rng.randint(-bound, bound)
        if (x, y) not in seen:
            pts.append([x, y])
            seen.add((x, y))
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. 4x4 grid at different scales
    for scale in [1, 2, 3, 5, 8, 12]:
        cfg = _grid_init(scale)
        if _valid_config(cfg):
            init_configs.append(cfg)
    
    # 2. Random configurations
    for bound in [100, 200, 300, 500]:
        for _ in range(5):
            cfg = _random_init(rng, bound)
            if _valid_config(cfg):
                init_configs.append(cfg)
    
    # 3. Structured: 4x4 grid with one point shifted
    for scale in [2, 3, 5]:
        base = _grid_init(scale)
        for shift_idx in range(16):
            for dx in [-2, -1, 1, 2]:
                for dy in [-2, -1, 1, 2]:
                    cfg = [list(p) for p in base]
                    cfg[shift_idx] = [cfg[shift_idx][0] + dx * scale, cfg[shift_idx][1] + dy * scale]
                    if _valid_config(cfg):
                        init_configs.append(cfg)
    
    # Evaluate and sort
    scored_configs = []
    for cfg in init_configs:
        s, valid = _score(cfg)
        if valid:
            scored_configs.append((s, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Run simulated annealing on top configurations
    top_k = min(8, len(scored_configs))
    for i in range(top_k):
        if _time_left() < 5.0:
            break
        s, cfg = scored_configs[i]
        if s > best_score:
            best_score = s
            best_pts = [list(p) for p in cfg]
        
        # Allocate time
        remaining = _time_left()
        alloc = remaining / (top_k - i + 1)
        alloc = max(2.0, min(alloc, 25.0))
        
        result, rs = _simulated_annealing(cfg, rng, alloc)
        if result is not None and rs > best_score:
            best_score = rs
            best_pts = result
    
    # Final: if nothing valid, return grid
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts
