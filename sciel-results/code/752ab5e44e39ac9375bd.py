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
    # Initial 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_lines(points)
    best_points = [p[:] for p in points]
    
    # Local Search
    # Time limit check is not strictly possible in pure python without time module, 
    # but we can limit iterations. 120s is generous.
    # Let's do 5000 iterations of random moves.
    
    import time
    start_time = time.time()
    limit = 110 # seconds
    
    for _ in range(100000):
        if time.time() - start_time > limit:
            break
            
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Move to a random nearby position
        # Radius 2
        new_x = old_x + random.randint(-2, 2)
        new_y = old_y + random.randint(-2, 2)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness (simple check)
        # We'll rely on is_valid which checks distinctness
        
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            score = count_lines(points)
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in points]
            else:
                # Revert
                points[idx] = [old_x, old_y]
        else:
            # Revert
            points[idx] = [old_x, old_y]
            
    return best_points
