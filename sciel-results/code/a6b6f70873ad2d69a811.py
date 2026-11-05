import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 320.0  # 10s margin for safety

    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_set = []
    best_len = 0
    
    # Use a few different strategies or just many random restarts
    # Random restarts are simple and effective for this kind of constraint satisfaction
    
    # We can also try a deterministic "spiral" or "grid scan" order first
    # but random is usually better for avoiding local minima in greedy set construction
    
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(12345 + r)
        # Shuffle a copy to not modify all_points if we wanted to reuse it, 
        # but all_points is just a list of tuples, shuffling it in place is fine if we don't need original order
        # Actually, let's shuffle a copy or just use random.sample if we want to be safe, 
        # but shuffling in place is O(N).
        
        # Let's create a shuffled list
        points_to_try = all_points[:]
        random.shuffle(points_to_try)
        
        selected = []
        
        for p in points_to_try:
            if time.time() > deadline:
                break
            
            px, py = p
            valid = True
            
            # Check against all pairs in selected
            k = len(selected)
            for i in range(k):
                ax, ay = selected[i]
                dx1 = px - ax
                dy1 = py - ay
                d1_sq = dx1*dx1 + dy1*dy1
                
                # If d1_sq is 0, it's the same point, but points are unique in all_points
                # so d1_sq > 0 always for distinct points.
                
                for j in range(i + 1, k):
                    bx, by = selected[j]
                    dx2 = px - bx
                    dy2 = py - by
                    d2_sq = dx2*dx2 + dy2*dy2
                    
                    dx_ab = ax - bx
                    dy_ab = ay - by
                    d_ab_sq = dx_ab*dx_ab + dy_ab*dy_ab
                    
                    if d1_sq == d2_sq or d1_sq == d_ab_sq or d2_sq == d_ab_sq:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                selected.append(p)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_set = selected[:]
            
    return best_set
# EVOLVE-BLOCK-END
