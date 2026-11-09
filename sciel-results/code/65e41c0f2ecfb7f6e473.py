"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def evaluate(points):
    """
    Returns (score, is_valid).
    Score is the number of lines with exactly 4 points.
    is_valid is False if any line has 5 or more points.
    """
    n = len(points)
    if n < 4:
        return 0, True
    
    line_map = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx_n = dx // g
                dy_n = dy // g
            else:
                dx_n, dy_n = dx, dy
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            nx = dy_n
            ny = -dx_n
            g2 = _gcd(abs(nx), abs(ny))
            if g2 > 0:
                nx = nx // g2
                ny = ny // g2
            
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            c = nx * x1 + ny * y1
            key = (nx, ny, c)
            
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    score = 0
    is_valid = True
    for key, pt_set in line_map.items():
        if len(pt_set) >= 5:
            is_valid = False
            break
        if len(pt_set) == 4:
            score += 1
            
    return score, is_valid

def construct():
    random.seed(42)
    
    best_score = 0
    best_points = None
    
    # Initial configurations
    init_configs = []
    
    # 1. 4x4 Grid
    init_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. Random points
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        init_configs.append(pts)
        
    for init_pts in init_configs:
        current_points = [p[:] for p in init_pts]
        current_score, current_valid = evaluate(current_points)
        
        if not current_valid:
            # Try to fix by randomizing a bit? Or just skip?
            # Let's just skip invalid starts to be safe, or randomize until valid.
            # Randomizing until valid might take time.
            # For now, if invalid, we just proceed and let SA fix it?
            # No, SA might get stuck in invalid states if we don't penalize.
            # Let's restart this config if invalid.
            continue
            
        best_local_score = current_score
        best_local_points = [p[:] for p in current_points]
        
        # Simulated Annealing
        T = 20.0
        T_min = 0.1
        cooling = 0.995
        max_iters = 10000
        
        for it in range(max_iters):
            if T < T_min:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Random move
            move_range = random.choice([3, 5, 10, 20])
            new_x = old_x + random.randint(-move_range, move_range)
            new_y = old_y + random.randint(-move_range, move_range)
            
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            if any(current_points[i] == [new_x, new_y] for i in range(16) if i != idx):
                continue
                
            current_points[idx] = [new_x, new_y]
            new_score, new_valid = evaluate(current_points)
            
            if not new_valid:
                # Invalid state: penalize heavily
                new_score = -1000 
                # Revert
                current_points[idx] = [old_x, old_y]
                continue
            
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
                
            T *= cooling
            
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = [p[:] for p in best_local_points]
            
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-END
