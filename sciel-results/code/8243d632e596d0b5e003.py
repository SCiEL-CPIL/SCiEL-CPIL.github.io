import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_set = []
    best_score = 0
    
    # Pre-generate all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # We will run multiple greedy attempts
    # Each attempt uses a different random order
    attempt_count = 0
    while time.time() < deadline:
        attempt_count += 1
        random.seed(attempt_count)
        
        points = all_points[:]
        random.shuffle(points)
        
        current_set = []
        # dist_sets[i] holds the set of squared distances from current_set[i] to all other points in current_set
        dist_sets = []
        
        for p in points:
            px, py = p
            # Check if p can be added
            valid = True
            new_dists = []
            
            # 1. Check isosceles at p: all distances from p to existing points must be distinct
            # 2. Check isosceles at existing points: for each q in set, dist(p, q) must not be in dist_sets[i]
            
            for i, q in enumerate(current_set):
                qx, qy = q
                d2 = (px - qx)**2 + (py - qy)**2
                
                # Check if this distance is already present for q (isosceles at q)
                if d2 in dist_sets[i]:
                    valid = False
                    break
                
                new_dists.append(d2)
            
            if valid:
                # Check if all new distances are distinct (isosceles at p)
                if len(new_dists) != len(set(new_dists)):
                    valid = False
            
            if valid:
                # Add p to the set
                current_set.append(p)
                new_dist_set = set(new_dists)
                dist_sets.append(new_dist_set)
                
                # Update existing dist_sets with the new distances
                for i in range(len(current_set) - 1):
                    q = current_set[i]
                    d2 = (px - q[0])**2 + (py - q[1])**2
                    dist_sets[i].add(d2)
        
        if len(current_set) > best_score:
            best_score = len(current_set)
            best_set = current_set[:]
            
            # If we find a very large set, we can try to optimize further or just continue
            # For now, just keep searching for better random orders
    
    return best_set
