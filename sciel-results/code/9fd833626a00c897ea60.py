import time
import math
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Directions for neighbors on triangular lattice
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # 1. Generate initial configuration: Core (R=2) + 13 contiguous points from Ring 3
    core = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                core.add((a, b))
    
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                ring3.append((a, b))
    
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    ring3.sort(key=angle)
    r3_len = len(ring3) # 18
    
    # Find best contiguous block of 13
    best_init_score = -1
    best_init_points = None
    
    for i in range(r3_len):
        subset = [ring3[(i + k) % r3_len] for k in range(13)]
        full_set = list(core) + subset
        
        if len(set(full_set)) != 32:
            continue
        
        edges = count_edges(full_set)
        if edges > best_init_score:
            best_init_score = edges
            best_init_points = set(full_set)
            
    current_points = list(best_init_points)
    current_set = set(current_points)
    current_score = best_init_score
    
    # 2. Local Search
    # We will try to swap points to increase edge count.
    # Neighbor candidates: Points not in current_set, but adjacent to some point in current_set.
    # We restrict search to points with hex_dist <= 4 to keep it compact.
    
    # Precompute all potential points in range
    all_potential_points = set()
    for a in range(-5, 6):
        for b in range(-5, 6):
            if hex_dist(a, b) <= 4:
                all_potential_points.add((a, b))
                
    max_iter = 5000
    no_improve_count = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate candidates for addition
        add_candidates = set()
        for p in current_set:
            for da, db in dirs:
                np = (p[0] + da, p[1] + db)
                if np not in current_set and np in all_potential_points:
                    add_candidates.add(np)
                    
        if not add_candidates:
            break
            
        # Generate candidates for removal
        # We prefer removing points with low degree (few connections)
        # But for now, just try all points in current_set
        # To speed up, we can sample or prioritize.
        # Given 32 points, trying all 32 * len(add_candidates) is small.
        
        improved = False
        
        # Sort add candidates by potential gain (heuristic: number of neighbors in current_set)
        def potential_gain(p):
            cnt = 0
            for da, db in dirs:
                if (p[0] + da, p[1] + db) in current_set:
                    cnt += 1
            return cnt
            
        sorted_adds = sorted(add_candidates, key=potential_gain, reverse=True)
        
        # Limit number of swaps to try per iteration to stay fast
        # Try top 10 add candidates against all 32 remove candidates
        for add_p in sorted_adds[:10]:
            for remove_p in current_set:
                # Try swap
                new_set = current_set.copy()
                new_set.remove(remove_p)
                new_set.add(add_p)
                
                if len(new_set) != 32:
                    continue
                
                new_points = list(new_set)
                new_score = count_edges(new_points)
                
                if new_score > current_score:
                    current_score = new_score
                    current_set = new_set
                    current_points = new_points
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
                
        if not improved:
            no_improve_count += 1
            # If we haven't improved in a while, maybe shuffle or break
            if no_improve_count > 100:
                break

    return [list(p) for p in current_points]
