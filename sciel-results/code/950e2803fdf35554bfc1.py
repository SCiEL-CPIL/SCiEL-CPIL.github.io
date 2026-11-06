import time
import random
from math import gcd

def count_ordinary_lines(points):
    """
    Counts the number of ordinary lines (lines containing exactly 2 points) 
    for a given set of points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            g = gcd(abs(dx), abs(dy))
            dx_red = dx // g
            dy_red = dy // g
            
            if dx_red < 0 or (dx_red == 0 and dy_red < 0):
                dx_red = -dx_red
                dy_red = -dy_red
            
            c = dy_red * x1 - dx_red * y1
            
            key = (dx_red, dy_red, c)
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add((x1, y1))
            lines[key].add((x2, y2))
    
    ordinary_count = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def is_valid(points):
    if len(points) != 12:
        return False
    if len(set(points)) != 12:
        return False
    # Check not all collinear
    if len(points) < 3:
        return True
    p1, p2, p3 = points[0], points[1], points[2]
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
        # Check if all others are on this line
        for p in points[3:]:
            x4, y4 = p
            if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
                return True
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_count = float('inf')
    
    # Initial configurations
    configs = []
    
    # 1. 3x4 Grid
    configs.append([(x, y) for x in range(4) for y in range(3)])
    
    # 2. Random configurations
    for _ in range(10):
        random_config = []
        while len(random_config) < 12:
            p = (random.randint(-20, 20), random.randint(-20, 20))
            if p not in random_config:
                random_config.append(p)
        configs.append(random_config)
        
    # 3. Pappus-like (if we can construct one)
    # Skip for now, rely on search
    
    for init_config in configs:
        if not is_valid(init_config):
            continue
            
        current_points = list(init_config)
        current_count = count_ordinary_lines(current_points)
        
        if current_count < best_count:
            best_count = current_count
            best_points = [list(p) for p in current_points]
            
        no_improvement_count = 0
        while time.time() < deadline:
            if no_improvement_count > 200:
                break
            
            moved = False
            # Try to move a single point
            idx = random.randint(0, 11)
            x, y = current_points[idx]
            
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    new_point = (new_x, new_y)
                    
                    if new_point in current_points:
                        continue
                    
                    new_config = list(current_points)
                    new_config[idx] = new_point
                    
                    if not is_valid(new_config):
                        continue
                        
                    new_count = count_ordinary_lines(new_config)
                    
                    if new_count < current_count:
                        current_points = new_config
                        current_count = new_count
                        moved = True
                        no_improvement_count = 0
                        if current_count < best_count:
                            best_count = current_count
                            best_points = [list(p) for p in current_points]
                        break
                if moved:
                    break
            
            if not moved:
                no_improvement_count += 1
                # Random restart strategy: replace 1-3 points
                if no_improvement_count % 50 == 0:
                    num_replacements = random.randint(1, 3)
                    indices = random.sample(range(12), num_replacements)
                    for idx in indices:
                        new_x = random.randint(-50, 50)
                        new_y = random.randint(-50, 50)
                        new_point = (new_x, new_y)
                        if new_point not in current_points:
                            current_points[idx] = new_point
                    current_count = count_ordinary_lines(current_points)
                    
    return best_points
