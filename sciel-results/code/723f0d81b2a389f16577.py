"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

def _canonical_line(p1, p2):
    """Return a canonical tuple (A, B, C) for the line passing through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    # Normalize by GCD
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0) # Should not happen for distinct points
    A //= g
    B //= g
    C //= g
    # Ensure a unique sign representation
    # First non-zero element should be positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(points):
    """
    Calculate the score: number of lines with exactly 4 points.
    Returns -1 if any line has >= 5 points (invalid).
    """
    line_counts = Counter()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            line = _canonical_line(points[i], points[j])
            line_counts[line] += 1
    
    score = 0
    for count in line_counts.values():
        if count >= 10: # C(5,2)=10 pairs for 5 points
            return -1
        if count == 6: # C(4,2)=6 pairs for 4 points
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety
    
    best_points = None
    best_score = -1
    
    # Initial candidates
    candidates = []
    
    # 1. 4x4 Grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    candidates.append(grid)
    
    # 2. Random configurations
    # We want points in a reasonable range to make collisions (collinearities) more likely but not too dense.
    # Range [-20, 20] or so is good for small integers.
    
    def generate_random_points():
        pts = set()
        while len(pts) < 16:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            pts.add((x, y))
        return [list(p) for p in pts]

    # Add a few random starts
    for _ in range(10):
        candidates.append(generate_random_points())

    def local_search(points):
        nonlocal best_points, best_score
        current_points = [p[:] for p in points]
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]

        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_point = current_points[idx]
            
            # Generate a new point
            # Strategy: try to move to a point that might form lines with existing ones
            # Simple random in a box
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            new_point = [new_x, new_y]
            
            # Check distinctness
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            current_points[idx] = new_point
            new_score = _score(current_points)
            
            if new_score == -1:
                # Invalid (5 collinear), revert
                current_points[idx] = old_point
                no_improve_count += 1
            elif new_score > current_score:
                # Improvement
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                no_improve_count = 0
            else:
                # No improvement or worse
                # Accept with small probability to escape local optima? 
                # Or just revert. Let's try a simple hill climb first.
                # If we want simulated annealing, we'd accept worse moves.
                # Given the small search space, hill climb might get stuck.
                # Let's add a small chance of accepting equal/worse moves if no_improve is high?
                # Actually, let's just revert for pure hill climbing, but restart if stuck.
                current_points[idx] = old_point
                no_improve_count += 1

        # If we broke due to no improvement, we could try a random restart or mutation
        # For now, the outer loop handles multiple starts.

    # Run local search on each candidate
    for candidate in candidates:
        if time.time() > deadline:
            break
        local_search(candidate)
        
        # If we found a high score, maybe do more focused search?
        # The loop above handles time.

    # Ensure we have a valid result
    if best_points is None or best_score < 0:
        # Fallback to grid if something went wrong
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-END
