import time
import random
import math

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    # Check for duplicates
    if len(set(tuple(p) for p in points)) != n:
        return -1
        
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
            
    # Check collinearity and count 4-point lines
    lines_found = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Count points on the line and check for 5+ collinear
            count = 0
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                # Cross product to check collinearity
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    count += 1
                    indices_on_line.add(k)
                    if count >= 5:
                        return -1
            
            if count == 4:
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
                
    return len(lines_found)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for 150s limit
    
    # Start with a 4x4 grid configuration which has many 4-point lines
    # A 4x4 grid has 4 horizontal, 4 vertical, and 2 diagonal lines with 4 points = 10 lines
    initial_points = [[x * 10, y * 10] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = _count_lines_exactly_4(best_points)
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Local search: try to improve by moving points
    stuck_counter = 0
    max_stuck = 500
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try different step sizes
        if random.random() < 0.05:
            # Very large jump
            new_x = old_x + random.randint(-50, 50)
            new_y = old_y + random.randint(-50, 50)
        elif random.random() < 0.15:
            # Large jump
            new_x = old_x + random.randint(-15, 15)
            new_y = old_y + random.randint(-15, 15)
        elif random.random() < 0.4:
            # Medium step
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            # Small step
            new_x = old_x + random.randint(-2, 2)
            new_y = old_y + random.randint(-2, 2)
        
        # Ensure bounds
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Avoid staying in place
        if new_x == old_x and new_y == old_y:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = _count_lines_exactly_4(current_points)
        
        if new_score > current_score:
            current_score = new_score
            stuck_counter = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Restart from best with some perturbation
                current_points = [p[:] for p in best_points]
                current_score = best_score
                stuck_counter = 0
                
                # Perturb a few points
                for _ in range(random.randint(2, 4)):
                    pidx = random.randint(0, 15)
                    current_points[pidx][0] += random.randint(-20, 20)
                    current_points[pidx][1] += random.randint(-20, 20)
                    current_points[pidx][0] = max(-1000, min(1000, current_points[pidx][0]))
                    current_points[pidx][1] = max(-1000, min(1000, current_points[pidx][1]))
                
                current_score = _count_lines_exactly_4(current_points)
                if current_score < best_score:
                    current_points = [p[:] for p in best_points]
                    current_score = best_score
                
    return best_points
