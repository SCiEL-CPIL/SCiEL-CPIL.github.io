import time
import random
import math

def _score(points):
    n = len(points)
    if n != 16:
        return -1
    
    # Check bounds and distinctness
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
        
    count = 0
    lines_seen = set()
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            nx = dx // g
            ny = dy // g
            
            if nx < 0 or (nx == 0 and ny < 0):
                nx = -nx
                ny = -ny
            
            if (x2, y2) < (x1, y1):
                ref_x, ref_y = x2, y2
            else:
                ref_x, ref_y = x1, y1
                
            line_key = (nx, ny, ref_x, ref_y)
            
            if line_key in lines_seen:
                continue
            lines_seen.add(line_key)
            
            cnt = 0
            for k in range(n):
                px, py = points[k]
                cross = (px - ref_x) * ny - (py - ref_y) * nx
                if cross == 0:
                    cnt += 1
                    if cnt > 4:
                        return -1
            
            if cnt == 4:
                count += 1
                
    return count

def _sa(initial_points, time_limit):
    current = [p[:] for p in initial_points]
    current_score = _score(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate move
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinct
        valid_move = True
        for i, p in enumerate(current):
            if i != idx and p[0] == new_x and p[1] == new_y:
                valid_move = False
                break
                
        if not valid_move:
            continue
            
        # Apply move
        old_p = current[idx][:]
        current[idx] = [new_x, new_y]
        
        new_score = _score(current)
        
        if new_score == -1:
            # Revert
            current[idx] = old_p
            # Penalize invalid moves heavily? Or just reject?
            # Rejecting is safer.
            T *= cooling
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = [p[:] for p in current]
        else:
            current[idx] = old_p
            
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Initial 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = grid
    best_score = _score(grid)
    
    # Run SA from grid
    p, s = _sa(grid, 20)
    if s > best_score:
        best_score = s
        best_points = p
        
    # Run SA from random
    random.seed(123)
    random_points = []
    for _ in range(16):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        random_points.append([x, y])
    # Ensure distinct
    unique = []
    for p in random_points:
        if p not in unique:
            unique.append(p)
    while len(unique) < 16:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if [x, y] not in unique:
            unique.append([x, y])
            
    p, s = _sa(unique, 20)
    if s > best_score:
        best_score = s
        best_points = p
        
    # Run SA from another structured config?
    # Maybe a 3x3 grid + 1 point?
    # Or just more random restarts if time permits.
    
    while time.time() < deadline:
        random_points = []
        for _ in range(16):
            x = random.randint(-200, 200)
            y = random.randint(-200, 200)
            random_points.append([x, y])
        unique = []
        for p in random_points:
            if p not in unique:
                unique.append(p)
        while len(unique) < 16:
            x = random.randint(-200, 200)
            y = random.randint(-200, 200)
            if [x, y] not in unique:
                unique.append([x, y])
                
        p, s = _sa(unique, 10)
        if s > best_score:
            best_score = s
            best_points = p
            
    return [[p[0], p[1]] for p in best_points]
