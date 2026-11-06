import time
import random
from itertools import combinations
from math import gcd

def _line_key(x1, y1, x2, y2):
    """Return a canonical key for the line through two integer points."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    if dx_n == 0:
        return ('v', x1)
    else:
        c = dy_n * x1 - dx_n * y1
        return ('s', dx_n, dy_n, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _point_on_line(px, py, x1, y1, x2, y2):
    """Check if point (px,py) lies on the line through (x1,y1) and (x2,y2)."""
    return (x2 - x1) * (py - y1) == (y2 - y1) * (px - x1)

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _try_create_collinearity(points, deadline):
    """Try to move a point onto a line defined by two other points to create a 3-collinearity."""
    best = [p[:] for p in points]
    best_score = _count_ordinary(points)
    current = [p[:] for p in points]
    current_score = best_score
    
    n = len(current)
    max_iters = 2000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        improved = False
        
        # For each pair of points, try to find a third point that can be moved onto their line
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = current[i]
                x2, y2 = current[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # For each other point k, try to move it onto the line through i,j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Check if point k is already on this line
                    if _point_on_line(current[k][0], current[k][1], x1, y1, x2, y2):
                        continue
                    
                    # Try moving point k to a position on the line through i,j
                    # The line is parametric: (x1 + t*dx_n, y1 + t*dy_n) for integer t
                    # Try several t values to find a valid new position
                    old_kx, old_ky = current[k]
                    for t in range(-5, 6):
                        if t == 0:
                            continue
                        new_x = x1 + t * dx_n
                        new_y = y1 + t * dy_n
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check for duplicates
                        is_dup = False
                        for m in range(n):
                            if m != k and current[m][0] == new_x and current[m][1] == new_y:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        # Check it's not the same as old position
                        if new_x == old_kx and new_y == old_ky:
                            continue
                        
                        current[k] = [new_x, new_y]
                        if not _not_all_collinear(current):
                            current[k] = [old_kx, old_ky]
                            continue
                        
                        new_score = _count_ordinary(current)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if new_score < best_score:
                                best_score = new_score
                                best = [p[:] for p in current]
                            break
                        else:
                            current[k] = [old_kx, old_ky]
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best, best_score

def _local_search(points, deadline):
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        for idx in range(12):
            old_x, old_y = current[idx]
            # Try nearby moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    continue
                break
            if improved:
                break
        if not improved:
            # Try collinearity creation
            if time.time() < deadline:
                coll_best, coll_score = _try_create_collinearity(current, deadline)
                if coll_score < current_score:
                    current = coll_best
                    current_score = coll_score
                    improved = True
                    if coll_score < best_score:
                        best_score = coll_score
                        best = [p[:] for p in current]
            
            if not improved:
                # Random restart within neighborhood
                idx = random.randint(0, 11)
                old_x, old_y = current[idx]
                for _ in range(80):
                    new_x = old_x + random.randint(-15, 15)
                    new_y = old_y + random.randint(-15, 15)
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                if not improved:
                    break
    
    return best, best_score

def construct():
    deadline = time.time() + 140
    
    random.seed(12345)
    
    # Multiple initial configurations
    initial_configs = []
    
    # 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Structured: two rows of 6
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Structured: 3 rows of 4
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Structured: points on a parabola y = x^2
    initial_configs.append([[x, x*x] for x in range(12)])
    
    # Structured: points on a circle (approximate with integers)
    circle_pts = []
    for x in range(-3, 4):
        for y in range(-3, 4):
            if x*x + y*y == 9 and [x,y] not in circle_pts:
                circle_pts.append([x, y])
    if len(circle_pts) >= 12:
        initial_configs.append(circle_pts[:12])
    
    # Structured: 2x6 with offsets to create more collinearities
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 2] for x in range(6)])
    
    # Structured: 4 points on y=0, 4 on y=1, 4 on y=2 with shifts
    initial_configs.append(
        [[x, 0] for x in range(4)] + 
        [[x, 1] for x in range(1, 5)] + 
        [[x, 2] for x in range(2, 6)]
    )
    
    # Random configs
    for _ in range(30):
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            initial_configs.append(pts)
    
    # Random configs with some structure (clusters)
    for _ in range(10):
        pts = []
        cx = random.randint(-5, 5)
        cy = random.randint(-5, 5)
        for _i in range(12):
            x = cx + random.randint(-3, 3)
            y = cy + random.randint(-3, 3)
            if [x, y] not in pts:
                pts.append([x, y])
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            initial_configs.append(pts)
    
    best_config = None
    best_score = float('inf')
    
    for cfg in initial_configs:
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    # Local search from best initial
    if best_config:
        best_config, best_score = _local_search(best_config, deadline)
    
    # Try local search from other good configs if time permits
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        score = _count_ordinary(cfg)
        if score <= best_score + 8:  # Only try promising ones
            cfg_best, cfg_score = _local_search(cfg, deadline)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    return best_config
