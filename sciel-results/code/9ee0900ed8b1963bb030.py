import random
import time
import math

def dist_sq(p1, p2):
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da*da + db*db + da*db

def is_valid(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if dist_sq(points[i], points[j]) < 1:
                return False
    return True

def edge_count(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if dist_sq(points[i], points[j]) == 1:
                count += 1
    return count

def greedy_seed(n):
    points = [(0, 0)]
    used = set(points)
    
    # Pre-calculate potential neighbors? No, just scan a growing box.
    # Max coord bound 64.
    # We will scan a box around the current centroid or just a fixed large box.
    # To be efficient, we can keep a list of "candidate" points.
    # Or just iterate over a box [-50, 50] x [-50, 50] each time? 
    # 100x100 = 10000 points. 32 steps. 320k checks. Each check O(N). 32*32*10000 ~ 10M ops. Fine.
    
    for _ in range(n-1):
        best_p = None
        best_score = -1
        # Scan a box. As the cluster grows, the box can stay fixed or grow.
        # Let's use a fixed box [-40, 40] to be safe.
        for a in range(-40, 41):
            for b in range(-40, 41):
                p = (a, b)
                if p in used:
                    continue
                # Check distance to all existing points
                valid = True
                score = 0
                for q in points:
                    d2 = dist_sq(p, q)
                    if d2 < 1:
                        valid = False
                        break
                    if d2 == 1:
                        score += 1
                if valid:
                    # Tie-breaker: minimize distance to centroid? Or just random?
                    # Prefer higher score.
                    if score > best_score:
                        best_score = score
                        best_p = p
                    elif score == best_score and best_p is not None:
                        # Random tie-break to avoid deterministic bias
                        if random.random() < 0.5:
                            best_p = p
        if best_p is None:
            break # Should not happen
        points.append(best_p)
        used.add(best_p)
    return points

def local_search(points, time_limit):
    current = [p for p in points]
    current_edges = edge_count(current)
    best = [p for p in current]
    best_edges = current_edges
    
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick a random point
        i = random.randint(0, len(current)-1)
        old_p = current[i]
        
        # Generate a new position for point i
        # Move to a random nearby point
        # Neighbors of old_p
        # Or just pick a random point in a box around old_p
        # Let's try moving to one of the 6 neighbors, or staying, or a random jump
        
        # Simple move: pick a random valid location in a small box around old_p
        # Or swap with another point?
        # Let's try: pick a random point j, swap i and j? 
        # Or move i to a random empty spot in the bounding box of the current set.
        
        # Let's use a "kick" move: move point i to a random lattice point 
        # within a radius R of its current position, ensuring it doesn't overlap 
        # (validity checked globally).
        
        new_a = old_p[0] + random.randint(-3, 3)
        new_b = old_p[1] + random.randint(-3, 3)
        new_p = (new_a, new_b)
        
        # Check if new_p is already occupied
        if new_p in current:
            continue
            
        # Tentatively move
        current[i] = new_p
        
        # Check validity
        valid = True
        for j in range(len(current)):
            if j == i:
                continue
            if dist_sq(current[i], current[j]) < 1:
                valid = False
                break
        
        if not valid:
            current[i] = old_p
            continue
            
        new_edges = edge_count(current)
        
        # Acceptance
        delta = new_edges - current_edges
        if delta > 0 or random.random() < math.exp(delta / T):
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best = [p for p in current]
        else:
            current[i] = old_p
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = []
    best_score = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        # Greedy seed
        seed = greedy_seed(32)
        if not is_valid(seed):
            continue
            
        # Local search
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Allocate time for local search
        ls_time = min(5.0, time_left - 0.5)
        
        points, score = local_search(seed, ls_time)
        
        if score > best_score:
            best_score = score
            best_points = points
            
        # Early exit if optimal
        if best_score == 76:
            break
            
    return [[p[0], p[1]] for p in best_points]
