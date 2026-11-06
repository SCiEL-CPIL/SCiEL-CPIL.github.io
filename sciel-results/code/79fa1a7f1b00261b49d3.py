import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    """
    Returns a normalized tuple (A, B, C) representing the line Ax + By + C = 0
    passing through p1 and p2.
    Normalization:
    1. A, B, C are integers.
    2. gcd(A, B, C) = 1.
    3. A > 0, or A == 0 and B > 0, or A == 0 and B == 0 and C > 0.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    # Avoid division by zero if points are identical (should not happen in valid sets)
    if A == 0 and B == 0:
        return None
        
    # Normalize sign
    # Make A positive. If A is 0, make B positive. If A and B are 0, make C positive.
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0:
        if B < 0:
            A, B, C = -A, -B, -C
        elif B == 0:
            if C < 0:
                A, B, C = -A, -B, -C
    
    # Normalize GCD
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    return (A, B, C)

def _score(points):
    """
    Returns the score: number of lines with exactly 4 points.
    Returns -1 if any line has 5 or more points, or if points are not distinct.
    """
    n = len(points)
    if n != 16:
        return -1
        
    # Check distinctness
    if len(set(tuple(p) for p in points)) < 16:
        return -1
        
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1
            
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] += 1
                
    score = 0
    for count in line_counts.values():
        if count >= 5:
            return -1
        if count == 4:
            score += 1
            
    return score

def _neighbor(points, step_size):
    """
    Generates a neighbor by perturbing one point's coordinates.
    """
    new_points = [p[:] for p in points]
    idx = random.randint(0, 15)
    # Perturb x or y
    if random.random() < 0.5:
        new_points[idx][0] += random.randint(-step_size, step_size)
    else:
        new_points[idx][1] += random.randint(-step_size, step_size)
        
    # Clamp to bounds to ensure validity check is fast (though score handles it)
    new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
    new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    
    return new_points

def _local_search(start_points, max_iters, temperature_start, temperature_end):
    """
    Simulated Annealing local search.
    """
    current = [p[:] for p in start_points]
    current_score = _score(current)
    
    best = [p[:] for p in current]
    best_score = current_score
    
    # If start is invalid, try to fix it by randomizing invalid points?
    # For simplicity, if start is invalid, we just proceed, but score is -1.
    # It's better to ensure start is valid.
    
    for i in range(max_iters):
        # Annealing schedule
        t = temperature_start * (temperature_end / temperature_start) ** (i / max_iters)
        
        # Step size can also be annealed, but fixed small step is often robust for this problem
        # Let's use a mix of small and medium steps
        step_size = random.choice([1, 1, 2, 3])
        
        neighbor = _neighbor(current, step_size)
        neighbor_score = _score(neighbor)
        
        if neighbor_score > current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = [p[:] for p in current]
                best_score = current_score
        elif neighbor_score == current_score:
            # Plateau: move with some probability to explore
            if random.random() < 0.3:
                current = neighbor
        else:
            # Worse score
            # If current_score is -1 (invalid) and neighbor is valid, accept
            if current_score < 0 and neighbor_score >= 0:
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = [p[:] for p in current]
                    best_score = current_score
            elif neighbor_score < 0:
                # Both invalid, reject
                pass
            else:
                # Both valid, neighbor worse
                delta = current_score - neighbor_score
                # Probability of accepting worse move
                if random.random() < 0.1: # Small chance to escape local optima even without annealing formula for simplicity/speed
                    current = neighbor
                    current_score = neighbor_score
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Initial candidate: 4x4 grid
    initial = [[x, y] for x in range(4) for y in range(4)]
    
    best_points = initial
    best_score = _score(best_points)
    
    # If initial is bad, try random
    if best_score < 0:
        # Generate random valid points
        while best_score < 0:
            pts = []
            seen = set()
            for _ in range(16):
                while True:
                    x = random.randint(-10, 10)
                    y = random.randint(-10, 10)
                    if (x, y) not in seen:
                        seen.add((x, y))
                        pts.append([x, y])
                        break
            best_points = pts
            best_score = _score(best_points)
    
    # List of starting points for multi-start
    starts = [best_points]
    
    # Add some structured perturbations of the grid
    for dx in [-2, -1, 1, 2]:
        for dy in [-2, -1, 1, 2]:
            perturbed = [p[:] for p in initial]
            # Move one corner
            perturbed[0][0] += dx
            perturbed[0][1] += dy
            starts.append(perturbed)
            
    # Add random starts
    for _ in range(5):
        pts = []
        seen = set()
        for _ in range(16):
            while True:
                x = random.randint(-20, 20)
                y = random.randint(-20, 20)
                if (x, y) not in seen:
                    seen.add((x, y))
                    pts.append([x, y])
                    break
        starts.append(pts)

    # Run local search from each start
    for start in starts:
        if time.time() > deadline:
            break
            
        # Determine iterations based on remaining time
        # Estimate: 1000 iterations takes ~0.1s? 
        # _score is ~120 gcd calls. Very fast.
        # Let's just run a fixed number of iterations per start, checking time.
        
        max_iters = 5000
        t_start = 2.0
        t_end = 0.1
        
        res_points, res_score = _local_search(start, max_iters, t_start, t_end)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_points
            
        # If we found a very good score (e.g., 12+), we might stop early or continue to polish
        # But we have time. Let's do a second pass on the best found so far.
        
    # Final polish: Run local search on the best found so far with more iterations
    if time.time() < deadline - 10:
        max_iters = 10000
        t_start = 1.0
        t_end = 0.01
        res_points, res_score = _local_search(best_points, max_iters, t_start, t_end)
        if res_score > best_score:
            best_score = res_score
            best_points = res_points

    return best_points

# EVOLVE-BLOCK-END
