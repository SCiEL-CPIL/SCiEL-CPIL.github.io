"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _try_structured_configs():
    """Try various structured configurations systematically."""
    configs = []
    
    # 4x4 grid minus 4 points (try all removals)
    grid = [(i, j) for i in range(4) for j in range(4)]
    for removed in combinations(range(16), 4):
        pts = [grid[i] for i in range(16) if i not in removed]
        configs.append(pts)
    
    # 3x4 grid (all 12 points)
    c_grid = [(i, j) for i in range(3) for j in range(4)]
    configs.append(c_grid)
    
    # 6x2 grid
    c_grid2 = [(i, j) for i in range(6) for j in range(2)]
    configs.append(c_grid2)
    
    # 4+4+4 on three parallel lines
    for offset in range(-3, 4):
        cfg = [[x, 0] for x in range(4)] + \
              [[x, 1] for x in range(4)] + \
              [[x + offset, 2] for x in range(4)]
        if len(set(tuple(p) for p in cfg)) == 12:
            configs.append(cfg)
    
    # 4+4+4 on three non-parallel lines
    c1 = [[0,0],[1,0],[2,0],[3,0],
          [1,1],[2,2],[3,3],[4,4],
          [5,0],[4,1],[3,2],[2,3]]
    if len(set(tuple(p) for p in c1)) == 12:
        configs.append(c1)
    
    best_points = None
    best_score = float('inf')
    
    for cfg in configs:
        if not _is_valid(cfg):
            continue
        s = _count_ordinary_fast(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
        if best_score <= 6:
            break
    
    return best_points, best_score

def _local_search(points, deadline):
    """
    Efficient local search with incremental scoring.
    Try single-point moves and two-point swaps.
    """
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Build candidate pool from lines through existing points
    coords = set()
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                g = 1
            dx //= g
            dy //= g
            # Extend line in both directions
            for t in range(-3, 4):
                coords.add((x1 + t*dx, y1 + t*dy))
    
    # Also add nearby points
    for p in current:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                coords.add((p[0]+dx, p[1]+dy))
    
    candidate_list = list(coords)
    random.shuffle(candidate_list)
    
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = tuple(current[i])
            for cand in candidate_list:
                if cand == old_pt:
                    continue
                if any(cand == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with structured configurations
    best_points, best_score = _try_structured_configs()
    
    # Local search from best configuration
    if time.time() < deadline - 10:
        ls_points, ls_score = _local_search(best_points, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If still above 6, try more random restarts with local search
    if best_score > 6 and time.time() < deadline - 15:
        random.seed(42)
        for restart in range(100):
            if time.time() >= deadline - 10:
                break
            
            # Generate random configuration on 3 lines
            pts = []
            # Line 1: y = 0
            x1s = random.sample(range(-5, 8), 4)
            for x in x1s:
                pts.append([x, 0])
            
            # Line 2: y = x + c
            c2 = random.randint(-3, 3)
            x2s = random.sample(range(-5, 8), 4)
            for x in x2s:
                pt = [x, x + c2]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    for x2 in range(-10, 10):
                        pt = [x2, x2 + c2]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            # Line 3: y = mx + b
            m3 = random.choice([-2,-1,1,2,3])
            b3 = random.randint(-3, 3)
            x3s = random.sample(range(-5, 8), 4)
            for x in x3s:
                pt = [x, m3 * x + b3]
                if all(tuple(pt) != tuple(p) for p in pts):
                    pts.append(pt)
                else:
                    for x3 in range(-10, 10):
                        pt = [x3, m3 * x3 + b3]
                        if all(tuple(pt) != tuple(p) for p in pts):
                            pts.append(pt)
                            break
            
            if len(pts) != 12 or not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in pts]
            
            # Local search
            if time.time() < deadline - 10:
                ls_points, ls_score = _local_search(pts, deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
