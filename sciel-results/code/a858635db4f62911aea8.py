import time
import random

def _count_ordinary_lines(points):
    n = len(points)
    ordinary = 0
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            is_ordinary = True
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x3 - x1) * dy == (y3 - y1) * dx:
                    is_ordinary = False
                    break
            if is_ordinary:
                ordinary += 1
    return ordinary

def _local_search(points, time_limit):
    current_points = [p[:] for p in points]
    current_cost = _count_ordinary_lines(current_points)
    start_time = time.time()
    best_points = [p[:] for p in current_points]
    best_cost = current_cost
    
    while time.time() - start_time < time_limit:
        idx = random.randint(0, len(current_points) - 1)
        orig_x, orig_y = current_points[idx]
        
        # Random move
        if random.random() < 0.2:
            new_x = random.randint(-15, 15)
            new_y = random.randint(-15, 15)
        else:
            new_x = orig_x + random.randint(-2, 2)
            new_y = orig_y + random.randint(-2, 2)
        
        # Check distinctness
        valid = True
        for p in current_points:
            if p is not current_points[idx]:
                if p[0] == new_x and p[1] == new_y:
                    valid = False
                    break
        if not valid:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_cost = _count_ordinary_lines(current_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            best_points = [p[:] for p in current_points]
            best_cost = new_cost
        else:
            current_points[idx] = [orig_x, orig_y]
            
        if best_cost == 6:
            break
            
    return best_points, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_cost = float('inf')
    
    while time.time() < deadline:
        points = []
        while len(points) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in points:
                points.append([x, y])
                
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 2:
            break
            
        search_time = max(0.5, remaining // 10)
        pts, cost = _local_search(points, search_time)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts
            if best_cost == 6:
                break
                
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
