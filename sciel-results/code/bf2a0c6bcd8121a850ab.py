import time
import random
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    # Points are (x, y)
    pts = [[x, y] for x in range(4) for y in range(4)]
    # Center them to allow moves in both directions
    pts = [[x - 2, y - 2] for x in range(4) for y in range(4)]
    
    best_score = 0
    best_pts = [p[:] for p in pts]
    current_score = 0
    current_pts = [p[:] for p in pts]
    
    def count_four_lines(points):
        n = len(points)
        # Use a set of normalized line representations to count unique lines?
        # No, the task asks for "number of lines containing exactly four of the points".
        # A line is defined by the set of points on it.
        # If 4 points are collinear, that's 1 line.
        # We need to count distinct lines that have exactly 4 points.
        
        lines = set()
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(dx, dy)
                if g == 0: continue # Should not happen for distinct points
                dx //= g
                dy //= g
                # Line equation: dy*x - dx*y = c
                c = dy * x1 - dx * y1
                # Normalize c? The line is unique by (dx, dy, c) if (dx,dy) is normalized.
                # But we need to check if exactly 4 points lie on it.
                # To avoid counting the same line multiple times from different pairs,
                # we can just collect all lines defined by pairs, then for each unique line, count points.
                # However, a line with 4 points is defined by $\binom{4}{2}=6$ pairs.
                # So we will see the same line 6 times.
                # We can store the line key in a set, then evaluate each unique line once.
                
                lines.add((dx, dy, c))
        
        score = 0
        for line in lines:
            dx, dy, c = line
            count = 0
            for x, y in points:
                if dy * x - dx * y == c:
                    count += 1
            if count == 4:
                score += 1
            elif count >= 5:
                return -1000 # Invalid configuration
                
        return score

    current_score = count_four_lines(current_pts)
    if current_score > best_score:
        best_score = current_score
        best_pts = [p[:] for p in current_pts]
        
    moves = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            moves.append((dx, dy))
            
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = current_pts[idx]
        
        # Pick a random move
        dx, dy = random.choice(moves)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        is_distinct = True
        for k in range(16):
            if k == idx:
                continue
            if current_pts[k][0] == new_x and current_pts[k][1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Apply move
        new_pts = [p[:] for p in current_pts]
        new_pts[idx] = [new_x, new_y]
        
        new_score = count_four_lines(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = [p[:] for p in new_pts]
            # Accept move
            current_pts = new_pts
            current_score = new_score
        elif new_score >= current_score and new_score != -1000:
            # Accept move to explore, but only if it doesn't reduce score significantly?
            # Hill climbing usually accepts if >= current.
            # However, if we get stuck, we might want to allow some worsening moves (simulated annealing).
            # For now, simple hill climbing with restarts if stuck?
            # Let's stick to accepting if >= current to move around plateaus.
            current_pts = new_pts
            current_score = new_score
        else:
            # Reject
            pass
            
        # Optional: If stuck for a while, try a larger jump or reset to best?
        # Simple strategy: If score hasn't improved in 1000 steps, reset to best and randomize?
        # Given time limit, simple hill climbing might suffice.
        
    return best_pts

