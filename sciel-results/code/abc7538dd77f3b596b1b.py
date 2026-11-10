import itertools
import time

def _count_ordinary(points):
    """Count the number of lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to count of points
    # Line key: normalized (A, B, C) for Ax + By + C = 0
    # To normalize: gcd(A, B, C) = 1, and (A, B) > (0, 0) or A == 0 and B > 0
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen
            
            # Make A positive, or if A==0 make B positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            # Divide by GCD
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    # Count lines with exactly 2 points
    ordinary_count = 0
    for count in line_counts.values():
        if count == 1: # 1 pair means 2 points
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Start with a 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    
    current_score = _count_ordinary(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search
    # Try moving each point to a nearby integer location
    # Limit search time to 100 seconds
    
    start_time = time.time()
    max_time = 100
    
    # Define neighbor moves
    # Small shifts
    moves = [
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ]
    
    # To avoid getting stuck, we can do a few iterations
    # But _count_ordinary is O(N^2). N=12, so 66 pairs. Very fast.
    # We can do many iterations.
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        for i in range(12):
            if (time.time() - start_time) > max_time:
                break
            x, y = points[i]
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                # Check bounds (loose, we can scale later if needed, but task says <= 10^9)
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                # Check distinctness
                new_points = [p[:] for p in points]
                new_points[i] = [nx, ny]
                
                # Check if point already exists
                if any(new_points[i] == p for j, p in enumerate(new_points) if j != i):
                    continue
                
                score = _count_ordinary(new_points)
                if score < current_score:
                    current_score = score
                    points = new_points
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                    break # Move to next point
    
    return best_points
