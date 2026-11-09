"""Search for 12 integer points minimizing the number of ordinary (exactly-2) lines."""
import time
import random
import math

_DEADLINE = 110.0  # safety margin under 150s
_start = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary(pts):
    """Exact count of lines containing exactly two of the given points."""
    n = len(pts)
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
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            if key in line_pts:
                line_pts[key].update((i, j))
            else:
                line_pts[key] = {i, j}
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
    p1 = pts[1]
    for i in range(2, 12):
        p2 = pts[i]
        if (p1[0]-p0[0])*(p2[1]-p0[1]) != (p1[1]-p0[1])*(p2[0]-p0[0]):
            return True
    return False

def _score(pts):
    return _count_ordinary(pts)

def _try_config(pts):
    pts = [[int(x), int(y)] for x, y in pts]
    if _valid(pts):
        return pts, _score(pts)
    return None, None

def _local_search_sa(start_pts, time_limit=10.0):
    """Run Simulated Annealing starting from start_pts."""
    rng = random.Random(hash(tuple(map(tuple, start_pts))) % 10000)
    cur_pts = [p[:] for p in start_pts]
    cur_val = _score(cur_pts)
    best_pts = [p[:] for p in cur_pts]
    best_val = cur_val
    
    start_time = time.time()
    T = 10.0
    cool_rate = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate a move
        idx = rng.randrange(12)
        # Move to a random point in a small neighborhood
        # Or swap with another point?
        # Let's try moving one point to a random location in a box
        # Box size depends on T?
        box_size = max(1, int(T))
        new_x = cur_pts[idx][0] + rng.randint(-box_size, box_size)
        new_y = cur_pts[idx][1] + rng.randint(-box_size, box_size)
        
        new_pts = [p[:] for p in cur_pts]
        new_pts[idx] = [new_x, new_y]
        
        # Check validity (distinctness)
        valid = True
        for i in range(12):
            if i == idx: continue
            if new_pts[i] == new_pts[idx]:
                valid = False
                break
        if not valid:
            continue
            
        new_val = _score(new_pts)
        delta = new_val - cur_val
        
        if delta < 0 or (T > 0.1 and rng.random() < math.exp(-delta / T)):
            cur_pts = new_pts
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_pts = [p[:] for p in cur_pts]
                
        T *= cool_rate
        if T < 0.01:
            break
            
    return best_pts, best_val

def _search():
    global _start
    _start = time.time()
    
    candidates = []
    
    # 1) 3x4 Grid
    grid_34 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_34)
    
    # 2) 4x3 Grid
    grid_43 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_43)
    
    # 3) 2x6 Grid
    grid_26 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_26)
    
    # 4) 6x2 Grid
    grid_62 = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(grid_62)
    
    # 5) Random points in small box
    rng = random.Random(42)
    for _ in range(10):
        pts = []
        seen = set()
        while len(pts) < 12:
            x = rng.randint(-5, 5)
            y = rng.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        candidates.append(pts)
        
    # 6) Specific known good configuration (if any found in past)
    # Try a "cross" or "star" pattern
    # Center (0,0), 4 on axes, 4 on diagonals, 4 more?
    # Let's just rely on SA from diverse starts.
    
    best_pts = None
    best_val = float('inf')
    
    # Allocate time to each candidate
    # We have ~110s. 10 candidates -> 10s each.
    # But some candidates might be better starts.
    # We will run SA on all, keeping the global best.
    
    for i, cand in enumerate(candidates):
        if time.time() - _start > _DEADLINE - 5:
            break
            
        # Validate start
        if not _valid(cand):
            continue
            
        # Run SA
        # Time limit per candidate: reduce as we go
        remaining = _DEADLINE - (time.time() - _start)
        t_limit = max(1.0, remaining / (len(candidates) - i))
        
        res_pts, res_val = _local_search_sa(cand, t_limit)
        
        if res_val < best_val:
            best_val = res_val
            best_pts = res_pts
            
        # Early termination if we hit the theoretical minimum
        if best_val == 6:
            break
            
    if best_pts is None:
        # Fallback to first valid candidate
        for cand in candidates:
            if _valid(cand):
                best_pts = cand
                best_val = _score(cand)
                break
    else:
        best_pts = best_pts
        
    return best_pts

def construct():
    global _start
    _start = time.time()
    try:
        res = _search()
        if res is None:
            res = [[x, y] for x in range(4) for y in range(3)]
        if not _valid(res):
            res = [[x, y] for x in range(4) for y in range(3)]
        return res
    except Exception:
        return [[x, y] for x in range(4) for y in range(3)]
