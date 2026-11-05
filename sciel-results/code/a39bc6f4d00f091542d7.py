"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(points):
    """
    Calculate the number of lines containing exactly 4 points.
    Also returns 0 if any line has 5 or more points (invalid).
    """
    n = len(points)
    # Map from line key to count of points
    line_counts = {}
    
    # Iterate over all pairs to define lines
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line equation: ax + by + c = 0
            # a = y1 - y2
            # b = x2 - x1
            # c = x1*y2 - x2*y1
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize the line equation to handle duplicates
            # Find gcd of a, b, c
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(a, b), c)
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                # This case (a=0, b=0) shouldn't happen for distinct points
                continue
                
            # Ensure a consistent sign, e.g., first non-zero is positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    # Now we have counts of how many pairs lie on each line.
    # If a line has k points, it contributes C(k, 2) to the count.
    # We need to invert this to find k for each line.
    # C(k, 2) = k(k-1)/2
    # Let count = C(k, 2). We solve for k.
    
    four_lines = 0
    for count in line_counts.values():
        # Find k such that k(k-1)/2 == count
        # k is integer.
        # 2*count = k^2 - k
        # k^2 - k - 2*count = 0
        # k = (1 + sqrt(1 + 8*count)) / 2
        val = 1 + 8 * count
        root = int(val ** 0.5)
        if root * root != val:
            # Should not happen if logic is correct, but safety check
            continue
        k = (1 + root) // 2
        if k == 4:
            four_lines += 1
        elif k >= 5:
            return 0 # Invalid configuration
            
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Shift to center? The grid is 0..3. 
    # Let's start with a random perturbation or just the grid.
    # The grid has 8 four-point lines.
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search
    n = 16
    # We will try to move one point at a time.
    
    while time.time() < deadline:
        # Pick a random point
        i = random.randint(0, n - 1)
        old_x, old_y = points[i]
        
        # Generate a random new position
        # Try small moves first, then larger
        if random.random() < 0.7:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            dx = random.randint(-100, 100)
            dy = random.randint(-100, 100)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        # Temporarily set the point
        points[i] = [new_x, new_y]
        
        # Check if it overlaps with any other point
        valid = True
        for j in range(n):
            if i == j:
                continue
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                valid = False
                break
                
        if not valid:
            points[i] = [old_x, old_y]
            continue
            
        # Calculate score
        score = _score(points)
        
        if score > current_score:
            current_score = score
            best_points = [p[:] for p in points]
            best_score = current_score
        else:
            # Revert
            points[i] = [old_x, old_y]
            
        # If we found a high score, we might want to explore around it more
        # Or just continue.
        
    return best_points
# EVOLVE-BLOCK-END
