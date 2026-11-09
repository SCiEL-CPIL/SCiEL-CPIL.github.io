import random
import math
import time
from math import gcd

def count_exact_4_lines(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return -1

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = -(x2 - x1)
        c = x1 * y2 - x2 * y1
        
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g == 0:
            return None
        a //= g
        b //= g
        c //= g
        
        if a < 0:
            a = -a
            b = -b
            c = -c
        elif a == 0:
            if b < 0:
                b = -b
                c = -c
            elif b == 0:
                if c < 0:
                    c = -c
        
        return (a, b, c)

    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k > 4:
            return -1 # Invalid
        if k == 4:
            count += 1
            
    return count

def construct():
    random.seed(42)
    
    # Start with 4x4 grid centered at 0
    # Points: -2, -1, 0, 1, 2 -> No, range(4) is 0,1,2,3.
    # Centered: -1.5, -0.5, 0.5, 1.5? Integers required.
    # Use -2, -1, 0, 1? Or 0,1,2,3?
    # 0,1,2,3 is fine.
    init_points = [[x, y] for x in range(4) for y in range(4)]
    
    current_points = [p[:] for p in init_points]
    current_score = count_exact_4_lines(current_points)
    if current_score < 0:
        current_score = 0 # Fallback logic if init is bad (it's not)
        
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # SA Parameters
    start_temp = 5.0
    end_temp = 0.01
    cooling_rate = 0.9995
    max_time = 110.0
    
    # Search Bounds
    min_coord = -10
    max_coord = 10
    
    start_time = time.time()
    temp = start_temp
    
    while temp > end_temp and (time.time() - start_time) < max_time:
        # Generate Neighbor
        # To avoid breaking the structure too much, we can pick a point 
        # and move it to a nearby integer coordinate.
        idx = random.randint(0, 15)
        ox, oy = current_points[idx]
        
        # Random step
        step_x = random.randint(-2, 2)
        step_y = random.randint(-2, 2)
        nx = ox + step_x
        ny = oy + step_y
        
        # If step is 0,0, try again or force a change?
        # Or just allow it (no change).
        
        # Keep within bounds? Not strictly required by task (|x|<=1000), 
        # but staying close to grid helps.
        if nx < min_coord or nx > max_coord or ny < min_coord or ny > max_coord:
             # If out of bounds, revert to original (no change) or clamp?
             # Let's just not apply the move if it goes too far out of our "search box"
             # But the task allows up to 1000.
             # For efficiency, we stay in [-10, 10].
             continue
             
        new_points = [p[:] for p in current_points]
        new_points[idx] = [nx, ny]
        
        new_score = count_exact_4_lines(new_points)
        
        # Handle invalid
        if new_score < 0:
            new_score = -10 # Penalty
        if current_score < 0:
            current_score = -10
            
        delta = new_score - current_score
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / temp):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
        temp *= cooling_rate

    # Final Verification
    final_score = count_exact_4_lines(best_points)
    if final_score < 0:
        # Return the initial grid if best is invalid
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
