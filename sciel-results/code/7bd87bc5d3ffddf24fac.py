import random
import time

def count_ordinary(points):
    n = len(points)
    count = 0
    # points is a list of (x, y)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            is_ordinary = True
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check if (x1, y1), (x2, y2), (x3, y3) are collinear
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    is_ordinary = False
                    break
            if is_ordinary:
                count += 1
    return count

def construct():
    # Start with 4x3 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    current_score = count_ordinary(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    
    while time.time() < deadline and T > 0.01:
        # Generate a neighbor
        new_points = [p[:] for p in points]
        idx = random.randint(0, 11)
        # Random move
        if random.random() < 0.5:
            new_points[idx][0] += random.randint(-5, 5)
        else:
            new_points[idx][1] += random.randint(-5, 5)
        
        # Ensure distinct points
        if len(set(map(tuple, new_points))) < 12:
            T *= cooling
            continue
            
        new_score = count_ordinary(new_points)
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                
        T *= cooling
        
    # If we found 6, we can stop early
    if best_score == 6:
        return best_points
        
    return best_points
