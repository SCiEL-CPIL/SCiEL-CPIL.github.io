"""
Program for the task: Twelve rational points with few ordinary lines.

Strategy:
1. Start with a small, hand-verified configuration that has a low number of ordinary lines.
2. Use a local search (hill climbing) with random restarts to try to improve the configuration.
3. The local search moves one point at a time to a random new position within the bounds,
   accepting the move only if it reduces the number of ordinary lines (or keeps it the same).
4. A time limit is enforced to ensure the program returns within the evaluation window.
"""

import time
import random
from math import gcd

# Global variables for the search
BEST_POINTS = None
BEST_SCORE = float('inf')
EVALUATION_LIMIT = 120  # 120 seconds
SAFETY_MARGIN = 5       # Leave 5 seconds of buffer

def _count_ordinary(points):
    """
    Count the number of ordinary lines (lines containing exactly two points)
    among the given list of points.
    
    points: list of (x, y) tuples
    Returns: int
    """
    n = len(points)
    if n < 3:
        return 0

    # Use a set to store unique lines.
    # A line is represented by its normal form: (A, B, C) such that Ax + By + C = 0
    # and gcd(A, B, C) = 1, with a canonical sign convention (e.g., A > 0 or A == 0 and B > 0).
    
    lines = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line equation: (y1 - y2)x - (x1 - x2)y + (x1*y2 - x2*y1) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                # This shouldn't happen for distinct points, but just in case
                continue
            A //= g
            B //= g
            C //= g
            
            # Canonical sign: A > 0, or A == 0 and B > 0
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
                
            lines.add((A, B, C))
            
    # Now, for each line, count how many points lie on it.
    # If the count is exactly 2, it's an ordinary line.
    ordinary_count = 0
    for line in lines:
        A, B, C = line
        cnt = 0
        for p in points:
            x, y = p
            if A*x + B*y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary_count += 1
            
    return ordinary_count

def _is_valid(points):
    """
    Check if the points are valid:
    - Exactly 12 distinct points
    - |x|, |y| <= 10^9
    - Not all collinear
    """
    if len(points) != 12:
        return False
    
    # Check bounds and distinctness
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check not all collinear
    if len(points) < 3:
        return True # Trivially not all collinear if <3 points, but we need 12
        
    # Pick 3 points to check if they are collinear.
    # If they are, check if the rest are also on that line.
    p1, p2, p3 = points[0], points[1], points[2]
    
    # Cross product of (p2-p1) and (p3-p1)
    cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
    if cross != 0:
        return True # Not collinear
        
    # If p1, p2, p3 are collinear, check if all points are on this line.
    # Line through p1, p2:
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1]
    
    for p in points[3:]:
        x, y = p
        if A*x + B*y + C != 0:
            return True # Found a point not on the line, so not all collinear
            
    return False # All points are on the same line

def _get_initial_points():
    """
    Return a set of 12 points that is known to have a low number of ordinary lines.
    
    A good candidate is a subset of a projective plane or a carefully chosen set.
    For 12 points, the theoretical minimum is 6.
    
    Let's try a 3x4 grid first, which is a standard example.
    A 3x4 grid has 12 points.
    Points: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)
    
    Let's calculate the number of ordinary lines for a 3x4 grid.
    There are 4 vertical lines (each with 3 points) -> 0 ordinary.
    There are 3 horizontal lines (each with 4 points) -> 0 ordinary.
    Diagonals:
    - Slope 1: (0,0)-(1,1)-(2,2) [3 pts], (0,1)-(1,2) [2 pts - ordinary], (1,0)-(2,1) [2 pts - ordinary], (2,0)-(3,1) [2 pts - ordinary], (3,0) ...
    This is getting complicated. Let's just use the _count_ordinary function on a few candidates.
    
    Candidate 1: 3x4 Grid
    [[0,0], [1,0], [2,0], [3,0],
     [0,1], [1,1], [2,1], [3,1],
     [0,2], [1,2], [2,2], [3,2]]
    
    Candidate 2: A set of points on a conic? No, that would be all collinear in projective sense? No.
    
    Let's just use the 3x4 grid as the starting point and let the search improve it.
    """
    return [(x, y) for x in range(4) for y in range(3)]

def construct():
    global BEST_POINTS, BEST_SCORE
    
    start_time = time.time()
    deadline = start_time + EVALUATION_LIMIT - SAFETY_MARGIN
    
    # Initialize with a good starting configuration
    initial_points = _get_initial_points()
    best_points = list(initial_points)
    best_score = _count_ordinary(best_points)
    
    BEST_POINTS = best_points
    BEST_SCORE = best_score
    
    if best_score == 6:
        # We reached the theoretical minimum
        return [[p[0], p[1]] for p in best_points]
        
    # Local Search
    # We will try random restarts and hill climbing.
    # Since the space is huge, we rely on the structure of the problem.
    # Points are integers, so we can try moving a point to a nearby integer coordinate.
    
    # To avoid getting stuck in local minima, we use simulated annealing or random restarts.
    # Given the 120s limit, we should be able to do a lot of steps.
    
    # A simple hill climbing with random restarts
    # We need to be careful about the step size. If we move points too far, we might break the structure.
    # But we are not constrained to be close to the initial points.
    
    # A better approach might be to start with a known optimal or near-optimal configuration.
    # For 12 points, is there a known configuration with 6 ordinary lines?
    # The orchard-planting problem is related.
    # For n=12, the maximum number of 3-point lines is 16 (for n=12, max 3-point lines is n(n-3)/6 = 12*9/6 = 18? No, that's for n divisible by 6? Let's check.)
    # Actually, the maximum number of ordinary lines is what we want to minimize.
    # Total lines = C(12,2) = 66.
    # Let L3 be the number of 3-point lines, L4 be the number of 4-point lines, etc.
    # Sum_{k>=2} k * Lk = 66
    # Sum_{k>=3} Lk = Number of lines with >=3 points.
    # Ordinary lines = L2.
    # We want to minimize L2.
    
    # For a 3x4 grid:
    # Vertical: 4 lines of 3 pts. L3 += 4.
    # Horizontal: 3 lines of 4 pts. L4 += 3.
    # Diagonals slope 1:
    #   (0,0)-(1,1)-(2,2): 3 pts. L3 += 1
    #   (0,1)-(1,2): 2 pts. L2 += 1
    #   (1,0)-(2,1): 2 pts. L2 += 1
    #   (2,0)-(3,1): 2 pts. L2 += 1
    #   (3,0): 1 pt.
    #   (0,2): 1 pt.
    # Diagonals slope -1:
    #   (0,2)-(1,1)-(2,0): 3 pts. L3 += 1
    #   (0,1)-(1,0): 2 pts. L2 += 1
    #   (1,2)-(2,1): 2 pts. L2 += 1
    #   (2,2)-(3,1): 2 pts. L2 += 1
    #   (3,2): 1 pt.
    #   (3,0): 1 pt.
    # Other diagonals (slope 2, 1/2, etc.)
    # This is tedious. Let's just run the code.
    
    # To make the search effective, we should move points in a way that preserves collinearity as much as possible.
    # A common technique is to move a point along a line defined by two other points, or to a position that creates a new 3-point line.
    
    # However, a simple random walk might be sufficient if we have enough time.
    # Let's try a strategy:
    # 1. Start with 3x4 grid.
    # 2. Repeatedly:
    #    a. Pick a random point i.
    #    b. Pick a new position (x, y) randomly from a range around its current position.
    #    c. If the new configuration is valid and has a lower score, accept it.
    #    d. If the new configuration has the same score, accept it with some probability (to allow plateau walking).
    # 3. If stuck, do a random restart from the initial configuration or a random valid configuration.
    
    # To generate a random valid configuration, we can just pick 12 random points in a box and check validity.
    
    max_steps = 1000000
    step_count = 0
    
    # We will use a temperature for simulated annealing
    temp = 10.0
    cooling_rate = 0.9999
    
    current_points = list(best_points)
    current_score = best_score
    
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = current_points[i]
        
        # Generate a new position
        # We want to explore the space effectively.
        # Let's try moving to a position that is close to the current one, but with some randomness.
        # Or, we can try to move it to a position that lies on a line with two other points.
        
        # Strategy 1: Move to a random nearby point.
        # Range: +/- 100
        new_x = old_x + random.randint(-100, 100)
        new_y = old_y + random.randint(-100, 100)
        
        # Ensure bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        new_points = list(current_points)
        new_points[i] = (new_x, new_y)
        
        if not _is_valid(new_points):
            continue
            
        new_score = _count_ordinary(new_points)
        
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points
                BEST_POINTS = best_points
                BEST_SCORE = best_score
                if best_score == 6:
                    break
        else:
            # Simulated annealing: accept worse moves with probability exp(-(delta)/temp)
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta/temp):
                current_points = new_points
                current_score = new_score
                
        temp *= cooling_rate
        if temp < 0.1:
            # Restart with a new random configuration
            # To generate a random configuration, pick 12 random points in a box [-100, 100]
            # and check if they are valid.
            attempts = 0
            while attempts < 100:
                random_points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(12)]
                if _is_valid(random_points):
                    random_score = _count_ordinary(random_points)
                    if random_score < best_score:
                        best_score = random_score
                        best_points = random_points
                        BEST_POINTS = best_points
                        BEST_SCORE = best_score
                        current_points = random_points
                        current_score = random_score
                    break
                attempts += 1
            temp = 10.0
            
    # Return the best points found
    return [[p[0], p[1]] for p in best_points]

# To ensure we don't time out, we wrap the call in a way that respects the deadline.
# The construct function itself has the deadline check.

# For testing purposes, we can run construct() if the module is executed directly.
if __name__ == "__main__":
    points = construct()
    print(points)
    print("Score:", _count_ordinary(points))
