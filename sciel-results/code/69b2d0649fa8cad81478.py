import random
import math
import time

# Map from pair count to number of points
PAIR_TO_POINTS = {
    1: 2, 3: 3, 6: 4, 10: 5, 15: 6, 21: 7, 28: 8, 36: 9, 45: 10,
    55: 11, 66: 12, 78: 13, 91: 14, 105: 15, 120: 16
}

def count_lines_and_check(points):
    n = len(points)
    line_pair_counts = {}
    is_valid = True
    
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            is_valid = False
        seen.add(p)
        
    if not is_valid:
        return 0, False

    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue

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
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    score = 0
    for key, pc in line_pair_counts.items():
        k = PAIR_TO_POINTS.get(pc)
        if k is None:
            # Should not happen for n=16, but just in case
            # If pc > 120, it's invalid? No, max pairs is 120.
            # If pc is not in map, it means k is not 2..16?
            # Actually, if pc is not in map, it's an error or k>16.
            # For n=16, max pc is 120.
            # If pc is e.g. 2, it's not a valid line of integer points?
            # No, every pair of points defines a line.
            # So pc must be one of the values in the map.
            # Wait, what if pc is not in the map?
            # e.g. pc=2? No line has exactly 2 pairs from 16 points?
            # A line with k points has k(k-1)/2 pairs.
            # If a line has 2 points, pc=1.
            # If a line has 3 points, pc=3.
            # It is impossible to have pc=2 for a set of points.
            # So if pc is not in map, something is wrong.
            # But we assume valid inputs.
            pass
            
        if k >= 5:
            is_valid = False
        elif k == 4:
            score += 1
            
    return score, is_valid

def construct():
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    current_score, current_valid = count_lines_and_check(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    if not current_valid:
        return best_points

    import time
    start_time = time.time()
    limit = 110.0
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() - start_time < limit:
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        if random.random() < 0.9:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        points[idx] = [new_x, new_y]
        
        new_score, new_valid = count_lines_and_check(points)
        
        if not new_valid:
            points[idx] = [old_x, old_y]
            continue
            
        if new_score > current_score:
            points[idx] = [new_x, new_y]
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            if T > min_T:
                delta_e = new_score - current_score
                if delta_e >= 0 or random.random() < math.exp(delta_e / T):
                    points[idx] = [new_x, new_y]
                    current_score = new_score
                else:
                    points[idx] = [old_x, old_y]
            else:
                points[idx] = [old_x, old_y]
                
        T *= cooling
        
        if random.random() < 0.01:
            T = 1.0
            
    return best_points
