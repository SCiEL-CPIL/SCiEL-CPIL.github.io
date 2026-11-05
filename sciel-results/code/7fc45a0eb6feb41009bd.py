import random
import time
import math
from collections import Counter

def construct():
    # Start with a 6x5 grid
    # Points in [-30, 30]^2
    # Grid is 0..5 x 0..4
    points = [(x, y) for x in range(6) for y in range(5)]
    
    # Helper to calculate area for a triple
    def area(p1, p2, p3):
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    
    # Calculate areas for triangles involving point i
    def get_areas_for_i(pts, i):
        pi = pts[i]
        areas = []
        for j in range(30):
            if j == i: continue
            pj = pts[j]
            for k in range(j+1, 30):
                if k == i: continue
                pk = pts[k]
                a = area(pi, pj, pk)
                areas.append(a)
        return areas

    # Initialize histogram
    hist = Counter()
    for i in range(30):
        for a in get_areas_for_i(points, i):
            hist[a] += 1
            
    # Current score is max count // 3
    current_max_count = max(hist.values()) // 3
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    
    best_points = points[:]
    best_score = current_max_count
    
    # To avoid getting stuck, we can occasionally restart T or do a larger move
    # But let's try standard SA first.
    
    while time.time() < deadline:
        # Pick random point
        i = random.randint(0, 29)
        
        # Pick random move
        # Mostly local, sometimes larger
        if random.random() < 0.8:
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)
        else:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            
        if dx == 0 and dy == 0:
            continue
            
        old_p = points[i]
        new_p = (old_p[0] + dx, old_p[1] + dy)
        
        # Check bounds and distinctness
        if not (-30 <= new_p[0] <= 30 and -30 <= new_p[1] <= 30):
            continue
        
        # Check distinctness
        # O(N) check
        is_distinct = True
        for p in points:
            if p == new_p:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Calculate old areas for i
        old_areas = get_areas_for_i(points, i)
        
        # Update points temporarily
        points[i] = new_p
        
        # Calculate new areas for i
        new_areas = get_areas_for_i(points, i)
        
        # Update histogram
        for a in old_areas:
            hist[a] -= 1
            if hist[a] == 0:
                del hist[a]
        for a in new_areas:
            hist[a] += 1
            
        # Calculate new score
        new_max_count = max(hist.values()) // 3
        
        # Acceptance
        if new_max_count > current_max_count:
            current_max_count = new_max_count
            if new_max_count > best_score:
                best_score = new_max_count
                best_points = points[:]
        else:
            delta = new_max_count - current_max_count
            if delta < 0:
                if random.random() < math.exp(delta / T):
                    current_max_count = new_max_count
                else:
                    # Revert
                    for a in new_areas:
                        hist[a] -= 1
                        if hist[a] == 0:
                            del hist[a]
                    for a in old_areas:
                        hist[a] += 1
                    points[i] = old_p
            else:
                # delta == 0
                # Accept with probability 0.5 to allow wandering
                if random.random() < 0.5:
                    current_max_count = new_max_count
                else:
                    for a in new_areas:
                        hist[a] -= 1
                        if hist[a] == 0:
                            del hist[a]
                    for a in old_areas:
                        hist[a] += 1
                    points[i] = old_p
        
        T *= cooling
        if T < T_min:
            T = T_min

    return best_points
