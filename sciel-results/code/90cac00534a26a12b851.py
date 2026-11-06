import time
import random

def count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    # Store lines as frozensets of point indices? Or just count per pair?
    # A line is ordinary if it contains exactly 2 points from the set.
    # We can iterate over all pairs, find the line, and count how many points are on it.
    # If count == 2, it's an ordinary line.
    # But each ordinary line is counted once per pair? No, an ordinary line has only 1 pair.
    # So we just need to check each pair.
    # If the line defined by pair (i, j) contains no other point k, it is an ordinary line.
    
    ordinary_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            # Check if any other point is on the line p1-p2
            is_ordinary = True
            for k in range(n):
                if k == i or k == j:
                    continue
                p3 = points[k]
                # Collinearity check: (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
                if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0]):
                    is_ordinary = False
                    break
            if is_ordinary:
                ordinary_count += 1
                
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 3x4 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    best_pts = [list(p) for p in pts]
    best_score = count_ordinary_lines(best_pts)
    
    # Local Search / Simulated Annealing
    current_pts = [list(p) for p in pts]
    current_score = best_score
    
    # Define a search space for moves
    # We can move a point to any integer coordinate in a range
    # Or just swap, or small shifts.
    # Small shifts are good for local optima.
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Try a random move
        # Step size decreases? Or fixed?
        # Let's try a random position in a box around the current point
        # Or a random position in a global box to allow jumps?
        # Global jumps help escape local minima.
        
        # Mix: 50% local move, 50% global move
        if random.random() < 0.5:
            # Local move
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        else:
            # Global move in a reasonable box
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
        
        # Ensure distinctness
        new_point = [new_x, new_y]
        if any(new_point == p for p in current_pts):
            continue
            
        # Evaluate
        current_pts[idx] = new_point
        new_score = count_ordinary_lines(current_pts)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_pts = [list(p) for p in current_pts]
        else:
            # Accept with probability exp(-delta/temp)
            delta = new_score - current_score
            if random.random() < pow(2.718, -delta / temp):
                current_score = new_score
            else:
                # Revert
                current_pts[idx] = [old_x, old_y]
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return best_pts
