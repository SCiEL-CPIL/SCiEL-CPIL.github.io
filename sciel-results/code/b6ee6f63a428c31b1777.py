"""Rewritten program: optimized local search with bitmask scoring."""
import time
import random
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_TIME_LIMIT = 110.0  # safety margin

def _time_left():
    if _START_TIME is None:
        return _TIME_LIMIT
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _score(points):
    """
    Returns (num_exact4_lines, valid).
    Uses bitmask to efficiently count points per line.
    """
    n = 16
    line_masks = {}
    
    # Pre-compute point coordinates for speed
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
            if dx == 0 and dy == 0:
                continue # Should not happen if distinct
            
            g = math.gcd(dx, dy)
            if g < 0:
                g = -g
            dx_n = dx // g
            dy_n = dy // g
            
            # Canonical direction
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
            
            # Bitmask update
            mask = (1 << i) | (1 << j)
            if key in line_masks:
                line_masks[key] |= mask
            else:
                line_masks[key] = mask
                
    exact4 = 0
    for mask in line_masks.values():
        # Count bits
        cnt = mask.bit_count()
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            return 0, False # Invalid
            
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

def construct():
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1
    
    # Initial: 4x4 Grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    if _valid_config(grid):
        s, _ = _score(grid)
        best_pts = [list(p) for p in grid]
        best_score = s
        
    def try_improve(init_pts, iterations):
        nonlocal best_pts, best_score
        if not _valid_config(init_pts):
            return
        cur = [list(p) for p in init_pts]
        cur_score, _ = _score(cur)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur]
            
        for _ in range(iterations):
            if _time_left() < 1.0:
                return
            
            # Mutation
            cand = [list(p) for p in cur]
            i = rng.randint(0, 15)
            # Random small move
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            new_p = [cand[i][0] + dx, cand[i][1] + dy]
            
            if not _inbounds(new_p):
                # Try a different point if out of bounds
                new_p = [rng.randint(-100, 100), rng.randint(-100, 100)]
                if not _inbounds(new_p):
                    continue
                
            # Check distinctness quickly
            if any((new_p[0] == p[0] and new_p[1] == p[1]) and idx != i for idx, p in enumerate(cand)):
                continue
                
            cand[i] = new_p
            
            cs, valid = _score(cand)
            if not valid:
                continue
                
            # Acceptance criteria
            if cs > cur_score:
                cur = cand
                cur_score = cs
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in cur]
            elif cs == cur_score and rng.random() < 0.1:
                cur = cand
                cur_score = cs
            elif cs < cur_score and rng.random() < 0.01:
                cur = cand
                cur_score = cs
                
    # Run search
    restarts = 0
    while _time_left() > 2.0:
        restarts += 1
        # Diverse starts
        if restarts == 1:
            init = grid
        else:
            # Perturb best found
            init = [list(p) for p in best_pts]
            # Add random noise to a few points to escape local optima
            for _ in range(3):
                idx = rng.randint(0, 15)
                init[idx] = [init[idx][0] + rng.randint(-20, 20), init[idx][1] + rng.randint(-20, 20)]
                
        try_improve(init, 500)
        
    if best_pts is None:
        best_pts = grid
        
    return best_pts
# EVOLVE-BLOCK-END
