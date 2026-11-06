import time
import random

def count_ordinary_lines(points):
    n = len(points)
    ordinary = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            found_third = False
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    found_third = True
                    break
            if not found_third:
                ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_score = float('inf')
    best_points = None
    
    # Try 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    score = count_ordinary_lines(points)
    if score < best_score:
        best_score = score
        best_points = [p[:] for p in points]
    
    # Random restarts
    while time.time() < deadline:
        # Generate a random configuration
        points = []
        for _ in range(12):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if [x, y] not in points:
                points.append([x, y])
            else:
                # Retry if duplicate
                while True:
                    x = random.randint(-10, 10)
                    y = random.randint(-10, 10)
                    if [x, y] not in points:
                        points.append([x, y])
                        break
        
        score = count_ordinary_lines(points)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in points]
            if best_score <= 6:
                break
        
        # Local search on this configuration
        while time.time() < deadline:
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
            
            if any(points[i] == [new_x, new_y] for i in range(12) if i != idx):
                continue
                
            points[idx] = [new_x, new_y]
            current_score = count_ordinary_lines(points)
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                if best_score <= 6:
                    return best_points
            else:
                points[idx] = [old_x, old_y]
                
            # If we haven't improved in a while, break out of local search
            # (Simplification: just keep going until deadline)
            
    return best_points
