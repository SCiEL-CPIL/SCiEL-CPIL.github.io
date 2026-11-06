import random
import math
import time
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize direction vector
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None # Should not happen for distinct points
    a = dy // g
    b = -dx // g
    
    # Normalize sign
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        
    # Calculate c such that ax + by + c = 0
    c = -(a * x1 + b * y1)
    
    return (a, b, c)

def _evaluate(points):
    """
    Returns (score, is_valid).
    score: number of lines with exactly 4 points.
    is_valid: False if any line has >= 5 points.
    """
    n = len(points)
    line_map = {}
    
    # Map each line to the set of points on it
    # We only need to check pairs, but we must ensure we capture all points on the line.
    # Since N is small, we can just check all points against the line defined by a pair.
    
    # Optimization: 
    # Iterate over all pairs to find unique lines.
    # For each unique line, count how many of the 16 points lie on it.
    
    unique_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key:
                unique_lines.add(key)
                
    score = 0
    is_valid = True
    
    for key in unique_lines:
        a, b, c = key
        count = 0
        for p in points:
            x, y = p
            if a * x + b * y + c == 0:
                count += 1
        if count >= 5:
            is_valid = False
            # We can break early if we want, but we might want to know the score of invalid configs?
            # The task says "no five collinear" is a constraint. 
            # "Maximize the number of lines containing exactly four... with no five collinear."
            # So invalid configs should have score 0 or be rejected.
            # Let's return 0 for invalid.
            return 0, False
        elif count == 4:
            score += 1
            
    return score, True

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial configuration: 4x4 grid
    # This has 10 lines of 4 points (4 rows, 4 cols, 2 diags)
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Scale up to use more of the space? 
    # The grid is in [0,3]x[0,3]. 
    # Let's center it or scale it to avoid boundary issues, though boundaries are 1000.
    # Actually, small coordinates are fine.
    
    current_score, _ = _evaluate(points)
    best_score = current_score
    best_points = [p[:] for p in points]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.001
    
    # Pre-compute possible moves? No, random is fine.
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new temperature and maybe a different start?
            # Or just keep going with min_T (random walk)
            T = min_T
            
        # Generate neighbor
        new_points = [p[:] for p in points]
        idx = random.randint(0, 15)
        
        # Choose a new random location
        # To ensure we explore, pick from a large range
        # But we must avoid collisions.
        # A simple way: try a few times to find a free spot
        max_retries = 10
        moved = False
        for _ in range(max_retries):
            nx = random.randint(-1000, 1000)
            ny = random.randint(-1000, 1000)
            # Check if occupied
            # Since N=16, linear search is fast
            occupied = False
            for i, p in enumerate(new_points):
                if i != idx and p[0] == nx and p[1] == ny:
                    occupied = True
                    break
            if not occupied:
                new_points[idx] = [nx, ny]
                moved = True
                break
        
        if not moved:
            # Extremely unlikely, but if grid is full? No, grid is huge.
            continue
            
        new_score, is_valid = _evaluate(new_points)
        
        if not is_valid:
            # Penalize heavily
            delta = -1000
        else:
            delta = new_score - current_score
            
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                
        T *= cooling_rate
        
    return best_points

# EVOLVE-BLOCK-END
