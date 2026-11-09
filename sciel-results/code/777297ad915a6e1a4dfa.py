import time
import random
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(pts):
    """Exact count of lines containing exactly two of the given points."""
    n = len(pts)
    # Map each unordered pair to its line; count how many points lie on each line.
    line_pts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            # Canonical direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line: dy*x - dx*y = c
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    ordinary = 0
    for key, s in line_pts.items():
        if len(s) == 2:
            ordinary += 1
    return ordinary

def _valid(pts):
    if len(pts) != 12:
        return False
    seen = set()
    for x, y in pts:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # not all collinear
    p0 = pts[0]
    for i in range(1, 12):
        p1 = pts[1]
        p2 = pts[i]
        if (p1[0]-p0[0])*(p2[1]-p0[1]) != (p1[1]-p0[1])*(p2[0]-p0[0]):
            return True
    return False

def _generate_candidates(cur, idx, box=10):
    cands = set()
    # Random moves
    for _ in range(10):
        x = cur[idx][0] + random.randint(-box, box)
        y = cur[idx][1] + random.randint(-box, box)
        cands.add((x, y))
    
    # Line-based moves
    for j in range(12):
        if j == idx: continue
        for k in range(j+1, 12):
            if k == idx: continue
            x1, y1 = cur[j]
            x2, y2 = cur[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0: continue
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            # Try placing point on this line
            # We iterate t such that point is (x1 + t*dx, y1 + t*dy)
            # We want to keep coordinates reasonable
            # Estimate t range based on distance to cur[idx]
            # Or just try a small range of t
            for t in range(-4, 8):
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # Avoid duplicates with existing points
                pt = (nx, ny)
                if pt in [(p[0], p[1]) for p in cur]:
                    continue
                cands.add(pt)
                
    return [list(c) for c in cands]

def _local_search(initial_pts, time_limit):
    start_time = time.time()
    best_pts = initial_pts[:]
    best_val = _count_ordinary(best_pts)
    
    cur_pts = initial_pts[:]
    cur_val = best_val
    
    # Simple hill climbing with random restarts within the time limit
    # We don't use full SA to keep it simple and fast, but we can add some randomness
    # Actually, let's add a simple acceptance criterion: accept if better, or with small prob if worse?
    # For minimization, greedy is often sufficient if the landscape is not too rugged.
    # But to escape local minima, we can do random restarts.
    
    stall = 0
    while time.time() - start_time < time_limit:
        idx = random.randint(0, 11)
        cands = _generate_candidates(cur_pts, idx)
        
        best_cand = None
        best_cand_val = cur_val
        
        for c in cands:
            new_pts = cur_pts[:]
            new_pts[idx] = c
            if not _valid(new_pts):
                continue
            val = _count_ordinary(new_pts)
            if val < best_cand_val:
                best_cand_val = val
                best_cand = new_pts
        
        if best_cand is not None and best_cand_val < cur_val:
            cur_pts = best_cand
            cur_val = best_cand_val
            if cur_val < best_val:
                best_pts = cur_pts[:]
                best_val = cur_val
            stall = 0
        else:
            stall += 1
            # If stuck, try a random move to escape
            if stall > 50:
                # Random restart of current point
                idx = random.randint(0, 11)
                cur_pts[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
                if _valid(cur_pts):
                    cur_val = _count_ordinary(cur_pts)
                stall = 0
                
    return best_pts, best_val

def _search():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_overall = None
    best_val_overall = float('inf')
    
    # Multiple initial configurations
    configs = []
    # 1. 3x4 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 2. 4x3 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    # 3. 4x4 Grid with corners removed
    configs.append([[x, y] for x in range(4) for y in range(4) if not ((x==0 and y==0) or (x==3 and y==0) or (x==0 and y==3) or (x==3 and y==3))])
    # 4. Random
    for _ in range(5):
        configs.append([[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)])
        
    # Filter valid configs
    valid_configs = []
    for c in configs:
        if _valid(c):
            valid_configs.append(c)
            
    # If no valid configs, use default
    if not valid_configs:
        valid_configs = [[[x, y] for x in range(4) for y in range(3)]]
        
    # Run local search on each config, allocating time based on remaining time
    # To be safe, we run one search with a fixed time slice, then if time remains, run another.
    # But simpler: just run local search on the best initial config found by quick evaluation.
    
    # Quick evaluation of initial configs
    best_init = None
    best_init_val = float('inf')
    for c in valid_configs:
        v = _count_ordinary(c)
        if v < best_init_val:
            best_init_val = v
            best_init = c
            
    if best_init is None:
        best_init = valid_configs[0]
        best_init_val = _count_ordinary(best_init)
        
    # Run local search on the best initial config
    # We can also run it on the second best if time permits
    time_per_search = 50.0
    remaining = deadline - time.time()
    
    if remaining > time_per_search:
        pts, val = _local_search(best_init, time_per_search)
        if val < best_val_overall:
            best_val_overall = val
            best_overall = pts
            
        # If we have time, try the next best initial config
        remaining = deadline - time.time()
        if remaining > 20.0:
             # Find next best init
             inits_sorted = sorted(valid_configs, key=lambda c: _count_ordinary(c))
             if len(inits_sorted) > 1:
                 next_init = inits_sorted[1]
                 pts2, val2 = _local_search(next_init, min(remaining - 5.0, 30.0))
                 if val2 < best_val_overall:
                     best_val_overall = val2
                     best_overall = pts2

    if best_overall is None:
        best_overall = best_init
        best_val_overall = best_init_val
        
    return best_overall

def construct():
    try:
        res = _search()
        if not _valid(res):
            res = [[x, y] for x in range(4) for y in range(3)]
        return res
    except Exception:
        return [[x, y] for x in range(4) for y in range(3)]
