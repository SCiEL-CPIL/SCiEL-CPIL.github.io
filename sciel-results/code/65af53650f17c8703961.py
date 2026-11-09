import time
import random
from itertools import combinations
from collections import defaultdict

def _line_key(p1, p2):
    """Get canonical key for the line through two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, p1[0] * dy - p1[1] * dx)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    line_map = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].append(i)
    
    count = 0
    for key, indices in line_map.items():
        if len(indices) == 4:
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any line contains 5 or more points."""
    n = len(points)
    line_map = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_map[key].append(i)
    
    for key, indices in line_map.items():
        if len(indices) >= 5:
            return True
    return False

def _score(points):
    """Score: number of exact-4 lines, penalize if any 5+ collinear."""
    if _has_five_collinear(points):
        return -1000
    return _count_exact4(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Simulated annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    temperature = 50.0
    cooling = 0.9995
    
    random.seed(42)
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Generate a perturbation: move one point
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        # Random new position within bounds
        new_x = random.randint(-50, 50)
        new_y = random.randint(-50, 50)
        
        # Ensure distinctness
        if any(p[0] == new_x and p[1] == new_y for p in current_points if p is not current_points[idx]):
            continue
        
        current_points[idx] = [new_x, new_y]
        new_score = _score(current_points)
        
        # Accept if better, or with probability based on temperature
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta == 0:
                # Same score, accept with some probability
                if random.random() < 0.3:
                    pass  # keep new point
                else:
                    current_points[idx] = [old_x, old_y]
            else:
                accept_prob = min(1.0, 1.0 / (1.0 + delta))
                if random.random() < accept_prob:
                    current_score = new_score
                else:
                    current_points[idx] = [old_x, old_y]
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
        
        # Periodic restart from best
        if step % 5000 == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            temperature = 30.0
    
    return best_points
