import random
import time

def score(points):
    areas = {}
    n = len(points)
    # Optimization: local variables
    p = points
    for i in range(n):
        x1, y1 = p[i]
        for j in range(i+1, n):
            x2, y2 = p[j]
            for k in range(j+1, n):
                x3, y3 = p[k]
                d = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if d > 0:
                    areas[d] = areas.get(d, 0) + 1
    if not areas:
        return 0
    return max(areas.values())

def construct():
    start_time = time.time()
    timeout = 88
    
    # Initial Candidates
    cands = []
    # 1. Grid 6x5
    cands.append([(x, y) for x in range(6) for y in range(5)])
    # 2. Grid 5x6
    cands.append([(y, x) for x in range(6) for y in range(5)])
    # 3. Diamond subset
    diamond = []
    for r in range(5):
        for x in range(-r, r+1):
            for y in range(-r, r+1):
                if abs(x) + abs(y) == r:
                    diamond.append((x, y))
    random.shuffle(diamond)
    cands.append(diamond[:30])
    # 4. Random
    cands.append([(random.randint(-30, 30), random.randint(-30, 30)) for _ in range(30)])
    
    best_score = 0
    best_points = None
    
    for c in cands:
        # Ensure distinctness
        if len(set(c)) < 30:
             # Fix distinctness for random candidate
             pts = list(set(c))
             while len(pts) < 30:
                 pts.append((random.randint(-30, 30), random.randint(-30, 30)))
             c = pts
        s = score(c)
        if s > best_score:
            best_score = s
            best_points = c[:]
            
    if best_points is None:
        best_points = cands[0]
        
    # Local Search
    current_points = best_points[:]
    current_score = best_score
    point_set = set(current_points)
    
    stuck_counter = 0
    
    while time.time() - start_time < timeout:
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Generate new point
        new_x = random.randint(-30, 30)
        new_y = random.randint(-30, 30)
        
        if (new_x, new_y) in point_set:
            continue
            
        # Apply move
        current_points[idx] = (new_x, new_y)
        point_set.remove((old_x, old_y))
        point_set.add((new_x, new_y))
        
        new_score = score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            stuck_counter = 0
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            point_set.remove((new_x, new_y))
            point_set.add((old_x, old_y))
            stuck_counter += 1
            
            # If stuck for a long time, restart from a random config or seed
            if stuck_counter > 3000:
                # Try a new random start
                new_start = [(random.randint(-30, 30), random.randint(-30, 30)) for _ in range(30)]
                # Ensure distinct
                while len(set(new_start)) < 30:
                    new_start[random.randint(0, 29)] = (random.randint(-30, 30), random.randint(-30, 30))
                    
                current_points = new_start
                point_set = set(current_points)
                current_score = score(current_points)
                stuck_counter = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]
                    
    return best_points
