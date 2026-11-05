import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def _count_4lines(points):
    """
    Counts the number of lines containing exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n < 4:
        return 0
    
    # Map from line key to count of points
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            g = gcd(abs(dx), abs(dy))
            if g != 0:
                dx //= g
                dy //= g
            else:
                # This shouldn't happen for distinct points
                continue
                
            # Canonical form for line: Ax + By + C = 0
            # A = dy, B = -dx
            # C = -(A*x1 + B*y1)
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C) to ensure unique representation
            # Make A positive, or if A is 0, make B positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            # GCD of A, B, C might not be 1 if we just normalized direction.
            # Actually, since dx, dy are coprime, A and B are coprime.
            # C is determined. The triple (A, B, C) is unique for the line.
            key = (A, B, C)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
                
    # Note: The above counts pairs. 
    # If a line has k points, it contributes C(k, 2) to the pair count for that key.
    # We need to recover k from the pair count? No, that's ambiguous.
    # Better: iterate points and assign them to lines?
    
    # Alternative approach:
    # For each point, find lines through it.
    # Simpler: Just use the pair counting but map pair count to point count?
    # C(k,2) = k(k-1)/2.
    # k=2 -> 1 pair
    # k=3 -> 3 pairs
    # k=4 -> 6 pairs
    # k=5 -> 10 pairs
    
    # Let's re-evaluate line_counts values.
    # The value in line_counts[key] is the number of pairs on that line.
    # We can invert this to find k.
    
    four_lines = 0
    for key, pair_count in line_counts.items():
        # Solve k(k-1)/2 = pair_count for k
        # k^2 - k - 2*pair_count = 0
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        disc = 1 + 8 * pair_count
        sqrt_disc = int(disc ** 0.5)
        if sqrt_disc * sqrt_disc != disc:
            # Should not happen if points are integer and logic is correct
            continue
        k = (1 + sqrt_disc) // 2
        if k == 5:
            return -1 # Invalid
        if k == 4:
            four_lines += 1
            
    return four_lines

def _is_valid(points):
    """Check if points are distinct and no 5 collinear."""
    if len(points) != len(set(map(tuple, points))):
        return False
    return _count_4lines(points) != -1

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_4lines(best_points)
    
    # Local Search / Random Restart
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Pick a new random position
        new_x = random.randint(-1000, 1000)
        new_y = random.randint(-1000, 1000)
        
        # Ensure distinctness
        if any(p == [new_x, new_y] for p in current_points):
            continue
            
        # Temporarily move
        current_points[idx] = [new_x, new_y]
        
        # Check validity (no 5 collinear)
        score = _count_4lines(current_points)
        
        if score == -1:
            # Invalid, revert
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            continue
            
        # If valid, check score
        if score > current_score:
            current_score = score
            no_improve_count = 0
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in current_points]
        else:
            # Revert if not improving (strict hill climbing)
            # Or accept with probability in SA? 
            # For now, strict hill climbing with restarts.
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Random restart
            current_points = [p[:] for p in best_points]
            current_score = best_score
            no_improve_count = 0
            
    return best_points
# EVOLVE-BLOCK-END
