import time
import random
import math
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            if dx_n == 0:
                key = ('v', x1)
            else:
                c = dy_n * x1 - dx_n * y1
                key = ('s', dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

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

def _simulated_annealing(points, deadline):
    """
    Perform simulated annealing to minimize ordinary lines.
    Uses incremental updates for speed.
    """
    n = len(points)
    current = [p[:] for p in points]
    
    # Initialize line counts and ordinary count
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
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

    # Build initial state
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(current[i], current[j])
            if key:
                line_counts[key] = line_counts.get(key, 0) + 1
    
    current_ordinary = sum(1 for cnt in line_counts.values() if cnt == 1)
    best = [p[:] for p in current]
    best_ordinary = current_ordinary
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    start_time = time.time()
    
    while time.time() < deadline and T > min_T:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = current[idx]
        
        # Generate a random new position nearby
        # Scale step size with temperature?
        step = int(T * 10) + 1
        new_x = old_x + random.randint(-step, step)
        new_y = old_y + random.randint(-step, step)
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check for duplicates
        is_dup = False
        for j in range(n):
            if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Check collinearity of all points
        # We can skip this if we assume random moves rarely make all collinear, 
        # but we must enforce it. 
        # For speed, we can check after move.
        
        new_point = [new_x, new_y]
        
        # Calculate delta in ordinary lines
        delta = 0
        
        # 1. Remove old point
        # For each other point j, the line (old, j) loses a point
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(current[idx], current[j])
            if key is None:
                continue
            cnt = line_counts.get(key, 0)
            if cnt == 0:
                continue # Should not happen
            # The line loses 1 point
            if cnt == 2:
                # Becomes 1, so it was ordinary, now not. 
                # Wait, if cnt==2, it was counted as ordinary.
                # After removal, cnt becomes 1. 
                # A line with 1 point is not an ordinary line (needs exactly 2).
                # So we lose 1 ordinary line.
                delta -= 1
            elif cnt == 3:
                # Becomes 2, so it was not ordinary, now is.
                # We gain 1 ordinary line.
                delta += 1
            # Update line count
            line_counts[key] = cnt - 1
            if line_counts[key] == 0:
                del line_counts[key]
                
        # 2. Add new point
        # For each other point j, the line (new, j) gains a point
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(new_point, current[j])
            if key is None:
                continue
            cnt = line_counts.get(key, 0)
            # The line gains 1 point
            if cnt == 0:
                # Was 0, becomes 1. Not ordinary.
                pass
            elif cnt == 1:
                # Was 1, becomes 2. Becomes ordinary.
                # Wait, if cnt==1, it was NOT counted as ordinary (needs exactly 2).
                # Now it becomes 2, so it IS ordinary.
                # We gain 1 ordinary line.
                delta += 1
            elif cnt == 2:
                # Was 2, becomes 3. Was ordinary, now not.
                # We lose 1 ordinary line.
                delta -= 1
            # Update line count
            line_counts[key] = cnt + 1
            
        # Check if all collinear
        # This is expensive to check every time. 
        # But if we move a point, it's unlikely to make all 12 collinear unless we are in a specific state.
        # We can check occasionally or use a heuristic.
        # For safety, let's check.
        if not _not_all_collinear(current):
            # Revert changes
            # This is complex to revert easily. 
            # Better to check before applying.
            pass # Placeholder
            
        # Actually, checking collinearity is O(n). 
        # Let's just check it.
        current[idx] = new_point
        if not _not_all_collinear(current):
            # Revert
            current[idx] = [old_x, old_y]
            # Revert line counts? 
            # This is getting messy. 
            # Let's just reject the move if it makes them collinear.
            # But we already updated line_counts.
            # We need to rollback.
            # For simplicity and safety, let's just reject moves that result in all collinear.
            # Since we updated line_counts, we must undo.
            
            # Undo add
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(new_point, current[j])
                if key:
                    line_counts[key] -= 1
                    if line_counts[key] == 0:
                        del line_counts[key]
            
            # Undo remove
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(current[idx], current[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
            
            current[idx] = [old_x, old_y]
            continue
            
        new_ordinary = current_ordinary + delta
        
        # Acceptance probability
        if new_ordinary < current_ordinary:
            accept = True
        else:
            # Delta is positive (worse)
            prob = math.exp(-new_ordinary / T)
            accept = random.random() < prob
            
        if accept:
            current_ordinary = new_ordinary
            if current_ordinary < best_ordinary:
                best_ordinary = current_ordinary
                best = [p[:] for p in current]
        else:
            # Revert changes
            # Undo add
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(new_point, current[j])
                if key:
                    line_counts[key] -= 1
                    if line_counts[key] == 0:
                        del line_counts[key]
            
            # Undo remove
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(current[idx], current[j])
                if key:
                    line_counts[key] = line_counts.get(key, 0) + 1
                    
            current[idx] = [old_x, old_y]
            
        T *= cooling_rate
        
    return best, best_ordinary

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 3x4 Grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 Grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Random points
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            initial_configs.append(pts)
            
    # Try to find a configuration with low ordinary lines
    best_config = None
    best_score = float('inf')
    
    # Run SA on each config for a portion of the time
    time_per_config = (deadline - time.time()) / len(initial_configs)
    
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        cfg_deadline = time.time() + time_per_config
        # Ensure at least 1s per config
        if cfg_deadline < time.time() + 1:
            cfg_deadline = time.time() + 1
            
        res_config, res_score = _simulated_annealing(cfg, cfg_deadline)
        
        if res_score < best_score:
            best_score = res_score
            best_config = res_config
            
    # If we have time left, run SA on the best config found
    if best_config and time.time() < deadline - 5:
        res_config, res_score = _simulated_annealing(best_config, deadline - 2)
        if res_score < best_score:
            best_score = res_score
            best_config = res_config
            
    return best_config

