import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Target: 16 points
    N = 16
    max_coord = 1000
    
    def evaluate(points):
        """
        Returns a score. Higher is better.
        Penalizes 5+ collinearities heavily.
        Rewards exactly 4 collinearities.
        """
        if len(points) != N:
            return -1000000
        # Check distinctness
        if len(set(tuple(p) for p in points)) != N:
            return -1000000
        
        # Check bounds
        for p in points:
            if abs(p[0]) > max_coord or abs(p[1]) > max_coord:
                return -1000000
        
        score = 0
        # Count lines
        # We need to count unique lines.
        # A line is defined by a pair (i, j).
        # To avoid double counting, we only consider pairs i < j.
        # For each pair, determine the line equation or just check collinearity with others.
        
        # Efficient way:
        # For each pair (i, j), calculate the vector (dx, dy).
        # Normalize it? No, just check if other points k satisfy (k - i) is parallel to (j - i).
        # Cross product (kx - ix)*(jy - iy) - (ky - iy)*(jx - ix) == 0
        
        four_count = 0
        five_count = 0
        
        # We need to count unique 4-point lines.
        # A line with 4 points has C(4,2) = 6 pairs.
        # A line with 5 points has C(5,2) = 10 pairs.
        # We can just iterate all pairs, find the max collinear set for that line direction, 
        # but that's tricky with multiple lines.
        
        # Better: Iterate all pairs (i, j).
        # For each pair, count how many points lie on the line defined by i and j.
        # If count == 4, it's a candidate. But we must ensure we don't count the same 4-point line multiple times.
        # We can store the set of indices for each 4-point line found.
        
        lines_found = set()
        
        for i in range(N):
            xi, yi = points[i]
            for j in range(i + 1, N):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                
                # Count points on this line
                count = 2 # i and j
                indices_on_line = {i, j}
                
                for k in range(N):
                    if k == i or k == j:
                        continue
                    xk, yk = points[k]
                    # Cross product check
                    if (xk - xi) * dy - (yk - yi) * dx == 0:
                        count += 1
                        indices_on_line.add(k)
                
                if count >= 4:
                    # Normalize the line representation to use as a key
                    # The set of indices is a unique identifier for the group of collinear points
                    # But we only care about lines with EXACTLY 4 points.
                    # If count > 4, it's invalid (or penalized).
                    if count == 4:
                        # Add the frozenset of indices to lines_found
                        lines_found.add(frozenset(indices_on_line))
                    elif count > 4:
                        five_count += 1 # This will be overcounted, but we just need a penalty signal
                        # Actually, if there's a 5-point line, it's very bad.
                        # We can just return a very low score immediately if we detect 5+
                        # But we need to be careful: a 6-point line contains multiple 5-point subsets.
                        # For simplicity in local search, any 5+ is a hard constraint violation.
                        # We'll penalize based on the maximum count found.
                        # Let's just track if any 5+ exists.
        
        # Refine the counting to be accurate for the score
        # The above loop is O(N^3) which is 16^3 = 4096 ops per eval. Very fast.
        
        # However, the `five_count` logic above is flawed for exact scoring.
        # Let's restart the evaluation logic to be precise.
        
        max_collinear = 0
        exact_four_lines = set()
        
        for i in range(N):
            xi, yi = points[i]
            for j in range(i + 1, N):
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                
                # Count points on this line
                count = 2
                indices_on_line = {i, j}
                
                for k in range(N):
                    if k == i or k == j:
                        continue
                    xk, yk = points[k]
                    if (xk - xi) * dy - (yk - yi) * dx == 0:
                        count += 1
                        indices_on_line.add(k)
                
                if count > max_collinear:
                    max_collinear = count
                    
                if count == 4:
                    exact_four_lines.add(frozenset(indices_on_line))
        
        num_four = len(exact_four_lines)
        
        # Penalty for 5+ collinear
        if max_collinear >= 5:
            # Heavy penalty proportional to how many are collinear
            penalty = (max_collinear - 4) * 1000
            return num_four - penalty
        
        return num_four

    def random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        return pts

    best_score = -1000000
    best_points = None
    
    # Initial population: Try a few random starts and the grid
    # Grid 4x4 is a good starting point
    grid = [[x, y] for x in range(4) for y in range(4)]
    # Shift grid to center? Not necessary, bounds are large.
    # But grid has 10 lines.
    
    candidates = [grid]
    for _ in range(10):
        candidates.append(random_points())
        
    current_points = None
    current_score = -1000000
    
    # Local Search
    # We will perform a series of restarts
    # Within each restart, we do hill climbing
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        if restarts > 50: # Safety cap on restarts
            break
            
        if restarts == 1:
            p = grid
        else:
            p = random_points()
            
        p = [list(pt) for pt in p] # Deep copy
        s = evaluate(p)
        
        if s > best_score:
            best_score = s
            best_points = [list(pt) for pt in p]
            
        # Hill Climb
        no_improve_count = 0
        max_no_improve = 2000
        temp = 50.0
        cooling = 0.999
        
        while no_improve_count < max_no_improve and time.time() - start_time < time_limit:
            # Pick a random point to move
            idx = random.randint(0, N - 1)
            old_x, old_y = p[idx]
            
            # Move to a nearby random point
            # Random step size
            step = random.randint(1, 10)
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)
            
            # Keep within bounds
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            # Ensure distinctness (simple check)
            is_distinct = True
            for k in range(N):
                if k != idx and p[k] == [new_x, new_y]:
                    is_distinct = False
                    break
            
            if not is_distinct:
                no_improve_count += 1
                continue
                
            p[idx] = [new_x, new_y]
            new_s = evaluate(p)
            
            if new_s > s:
                s = new_s
                no_improve_count = 0
                if s > best_score:
                    best_score = s
                    best_points = [list(pt) for pt in p]
            else:
                # Simulated Annealing acceptance
                delta = new_s - s
                if delta < 0:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        s = new_s
                    else:
                        # Reject move
                        p[idx] = [old_x, old_y]
                else:
                    s = new_s
                    no_improve_count = 0
                    if s > best_score:
                        best_score = s
                        best_points = [list(pt) for pt in p]
                        
                no_improve_count += 1
                temp *= cooling
                
        # End of one restart/hill climb
        
    # Final check
    if best_points is None:
        return grid
        
    # Ensure the returned points are valid
    # The evaluator will check, but we can double check here
    if evaluate(best_points) < 0:
        return grid
        
    return best_points
