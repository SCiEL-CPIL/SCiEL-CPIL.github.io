import random
import time
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 3:
        return 0
    
    line_counts = defaultdict(int)
    # Map from (A, B, C) to count
    # Line equation: A*x + B*y + C = 0
    # Normalize (A, B, C) such that gcd(A, B, C) = 1 and A > 0 or (A == 0 and B > 0) or (A == 0 and B == 0 and C > 0)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # A = y1 - y2
            # B = x2 - x1
            # C = -(A*x1 + B*y1)
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize
            g = __import__('math').gcd(A, __import__('math').gcd(B, C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                # This case shouldn't happen for distinct points
                continue
                
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
                
            line_counts[(A, B, C)] += 1
            
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial random points in a small box
    # We will run several restarts
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Generate 12 random points
        points = []
        while len(points) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            p = (x, y)
            if p not in points:
                points.append(p)
                
        # Convert to list of lists for consistency
        points = [list(p) for p in points]
        
        # Initial count
        score = count_ordinary_lines(points)
        
        # Local search
        for step in range(500):
            if time.time() > deadline:
                break
                
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            
            # New random position
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            
            if new_x == old_x and new_y == old_y:
                continue
                
            # Check if new point collides with others
            new_p = (new_x, new_y)
            if any(new_p == (p[0], p[1]) for p in points if p is not points[i]):
                continue
                
            # Calculate new score
            # To be efficient, we can just recalculate the whole thing for simplicity in the first pass,
            # but 12 points is small, so O(N^2) is fine.
            # N=12, pairs = 66. Very fast.
            
            points[i] = [new_x, new_y]
            new_score = count_ordinary_lines(points)
            
            if new_score < score:
                score = new_score
            elif new_score == score and random.random() < 0.5:
                score = new_score
            else:
                # Revert
                points[i] = [old_x, old_y]
                
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in points]
            
            # If we found 6, we can stop early
            if best_score == 6:
                break
                
    if best_points is None:
        # Fallback to the original bad solution
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        
    return best_points
