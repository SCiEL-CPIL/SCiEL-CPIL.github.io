import random
import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial state: 4x4 grid
    # Points: (x, y) for x in 0..3, y in 0..3
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = _evaluate(initial_points)
    
    # Ensure best_score is valid (no 5 collinear)
    if best_score is None:
        best_score = 0
        # Fallback to a random valid config if grid fails (it shouldn't)
        best_points = _random_valid_config()
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing / Random Restart Hill Climbing
    # We will try random moves.
    # Move: Pick a random point index, replace with a random point in a local neighborhood or global random.
    
    num_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and num_restarts < max_restarts:
        # If stuck, restart with a new random configuration or slight perturbation of best
        if num_restarts > 0:
            if random.random() < 0.5:
                current_points = _random_valid_config()
                current_score = _evaluate(current_points)
                if current_score is None:
                    current_score = 0
            else:
                # Perturb best
                current_points = [p[:] for p in best_points]
                # Randomly change 1-3 points
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 15)
                    current_points[idx] = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
                current_score = _evaluate(current_points)
                if current_score is None:
                    current_score = 0
        
        # Hill climbing loop
        while time.time() < deadline:
            # Generate a move
            idx = random.randint(0, 15)
            old_point = current_points[idx]
            
            # Strategy: Local search around current point or global random
            if random.random() < 0.8:
                # Local perturbation
                new_x = old_point[0] + random.randint(-5, 5)
                new_y = old_point[1] + random.randint(-5, 5)
            else:
                # Global random
                new_x = random.randint(-1000, 1000)
                new_y = random.randint(-1000, 1000)
            
            # Clamp to bounds
            new_x = max(-1000, min(1000, new_x))
            new_y = max(-1000, min(1000, new_y))
            
            new_point = [new_x, new_y]
            
            # Check distinctness
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            # Apply move
            current_points[idx] = new_point
            
            new_score = _evaluate(current_points)
            
            if new_score is not None and new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert if not improved (or invalid)
                current_points[idx] = old_point
                
            # Check deadline frequently
            if time.time() > deadline - 5.0:
                break
                
        num_restarts += 1
        
    return best_points

def _evaluate(points):
    """
    Returns the number of lines with exactly 4 points, or None if any line has >= 5 points.
    """
    n = len(points)
    if n != 16:
        return None
        
    # Map line coefficients to count of pairs on that line
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            # Canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    four_count = 0
    for count in line_counts.values():
        # count is number of pairs, which is k*(k-1)/2
        # k=4 => 6 pairs
        # k=5 => 10 pairs
        if count >= 10:
            return None
        if count == 6:
            four_count += 1
            
    return four_count

def _random_valid_config():
    """Generate a random configuration with no 5 collinear."""
    while True:
        points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
        # Ensure distinct
        if len(set(tuple(p) for p in points)) == 16:
            if _evaluate(points) is not None:
                return points

# Helper for math.gcd if not available (Python 3.5+ has it)
if not hasattr(math, 'gcd'):
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    math.gcd = _gcd
