import random
from math import gcd

def count_ordinary(points):
    n = len(points)
    ordinary = 0
    lines_seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in lines_seen:
                continue
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if a*xk + b*yk == c:
                    count += 1
            lines_seen.add(key)
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    best_points = None
    best_score = float('inf')
    
    # Define a set of initial configurations (grids and near-grids)
    # 1. 3x4 Grid
    # 2. 4x3 Grid
    # 3. Perturbed Grids
    
    def make_grid(rows, cols, start_x=0, start_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                pts.append([start_x + c, start_y + r])
        return pts

    def local_search_deterministic(points):
        points = [p[:] for p in points]
        improved = True
        # Limit iterations to avoid infinite loops, though convergence is fast
        while improved:
            improved = False
            for i in range(12):
                old_x, old_y = points[i]
                best_move = None
                best_move_score = count_ordinary(points)
                
                # Try all neighbors in a small box
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check distinctness
                        is_distinct = True
                        for k in range(12):
                            if k == i:
                                continue
                            if points[k][0] == new_x and points[k][1] == new_y:
                                is_distinct = False
                                break
                        if not is_distinct:
                            continue
                        
                        # Temporarily move
                        points[i] = [new_x, new_y]
                        score = count_ordinary(points)
                        
                        if score < best_move_score:
                            best_move_score = score
                            best_move = [new_x, new_y]
                        
                        # Restore
                        points[i] = [old_x, old_y]
                
                if best_move:
                    points[i] = best_move
                    improved = True
        
        return points, count_ordinary(points)

    # Initialize with a few structured configurations
    configs = []
    
    # 3x4 Grid
    configs.append(make_grid(3, 4))
    # 4x3 Grid
    configs.append(make_grid(4, 3))
    
    # 2x6 Grid
    configs.append(make_grid(2, 6))
    
    # 6x2 Grid
    configs.append(make_grid(6, 2))
    
    # Parabolic arc
    configs.append([[i, i*i] for i in range(12)])
    
    # Sine wave
    configs.append([[i, (i*i)%7] for i in range(12)])
    
    for config in configs:
        # Ensure distinct
        if len(set(tuple(p) for p in config)) < 12:
            continue
            
        optimized_points, score = local_search_deterministic(config)
        
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in optimized_points]
            
        if best_score <= 6:
            break
            
    return best_points
