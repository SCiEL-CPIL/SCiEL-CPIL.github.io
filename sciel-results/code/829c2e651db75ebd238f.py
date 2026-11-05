import time
import random

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    # Generate hexagon of radius 3 (37 points)
    R = 3
    hex_points = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                hex_points.append((a, b))
    
    # Start with known good removal set from inspiration
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    initial_points = [p for p in hex_points if p not in remove_set]
    
    if len(initial_points) != 32:
        # Fallback: just take first 32
        initial_points = hex_points[:32]
    
    initial_score = count_edges(initial_points)
    
    best_score = initial_score
    best_set = set(initial_points)
    
    # Generate a larger pool for swap candidates
    R_pool = 5
    pool = []
    for a in range(-R_pool, R_pool+1):
        for b in range(-R_pool, R_pool+1):
            if abs(a+b) <= R_pool:
                pool.append((a, b))
    
    pool_set = set(pool)
    
    # Local search: try swaps
    current_set = set(initial_points)
    current_score = initial_score
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        improved = False
        
        # Try all possible swaps
        best_swap_gain = 0
        best_swap_in = None
        best_swap_out = None
        
        points_list = list(current_set)
        
        for p_out in points_list:
            # Count edges lost if we remove p_out
            edges_lost = 0
            for p_other in points_list:
                if p_other != p_out and dist_sq(p_out, p_other) == 1:
                    edges_lost += 1
            
            # Try adding each point from pool not in current_set
            for p_in in pool:
                if p_in in current_set:
                    continue
                
                # Count edges gained if we add p_in
                edges_gained = 0
                for p_other in points_list:
                    if p_other != p_out and dist_sq(p_in, p_other) == 1:
                        edges_gained += 1
                
                gain = edges_gained - edges_lost
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap_in = p_in
                    best_swap_out = p_out
        
        if best_swap_gain > 0 and best_swap_in is not None:
            current_set.remove(best_swap_out)
            current_set.add(best_swap_in)
            current_score += best_swap_gain
            improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            # No improvement, try random restarts
            if iteration % 50 == 0 and iteration > 0:
                # Random perturbation: swap 2 random points
                if len(current_set) >= 2:
                    pts = list(current_set)
                    idx1, idx2 = random.sample(range(len(pts)), 2)
                    p_remove = pts[idx1]
                    # Find a random pool point not in set
                    candidates = [p for p in pool if p not in current_set]
                    if candidates:
                        p_add = random.choice(candidates)
                        current_set.remove(p_remove)
                        current_set.add(p_add)
                        # Recalculate score
                        current_score = count_edges(list(current_set))
    
    result = [[a, b] for a, b in best_set]
    
    # Verify
    if len(result) != 32:
        return [[a, b] for a in range(16) for b in range(2)]
    
    return result
