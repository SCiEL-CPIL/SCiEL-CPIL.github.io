import math
import random
import itertools

def count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    # Use a set of normalized line representations
    # A line can be represented by (A, B, C) for Ax + By + C = 0
    # Normalization: gcd(A, B, C) = 1, A > 0 or (A == 0 and B > 0)
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            # Find gcd of A, B, C
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical form
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    # Count lines with exactly 4 points
    count = 0
    for key, pts in line_counts.items():
        if len(pts) == 4:
            count += 1
            
    return count

def is_valid(points):
    """
    Check if points are distinct and no 5 are collinear.
    """
    if len(set(map(tuple, points))) != len(points):
        return False
        
    # Check for 5 collinear
    # Reuse line logic
    n = len(points)
    line_counts = {}
    import math
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            return False
            
    return True

def construct():
    # Start with the 4x4 grid which is a known good local optimum
    # This ensures we start with a high score (10) and can try to improve.
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = count_lines(points)
    
    # Simulated Annealing parameters
    start_temp = 5.0
    end_temp = 0.001
    cooling_rate = 0.9995
    current_temp = start_temp
    
    import time
    start_time = time.time()
    limit = 110 # seconds
    
    current_score = best_score
    current_points = [p[:] for p in points]
    
    while time.time() - start_time < limit:
        if current_temp < end_temp:
            # If temperature is very low, we are effectively doing hill climbing.
            # We can restart the annealing process or just let it run.
            # Let's restart with a higher temp to escape potential local minima.
            current_temp = start_temp * 0.5
            # Optional: reset current points to best points to restart from best
            # current_points = [p[:] for p in best_points]
            # current_score = best_score
            pass
            
        # Generate a neighbor
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random move step
        # Use a mix of small and large jumps
        if random.random() < 0.1:
            step_range = 20
        else:
            step_range = 5
            
        new_x = old_x + random.randint(-step_range, step_range)
        new_y = old_y + random.randint(-step_range, step_range)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_points = [p[:] for p in current_points]
        new_points[idx] = [new_x, new_y]
        
        if not is_valid(new_points):
            continue
            
        new_score = count_lines(new_points)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta > 0:
            # Accept better solution
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept worse solution with probability exp(delta / temp)
            # delta is negative
            # Avoid division by zero if temp is 0
            if current_temp > 0:
                prob = math.exp(delta / current_temp)
            else:
                prob = 0.0
            if random.random() < prob:
                current_points = new_points
                current_score = new_score
                
        current_temp *= cooling_rate
        
    return best_points
