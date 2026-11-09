import time
import random
import math
from collections import defaultdict

def _canonical_line(p1, p2):
    """
    Returns a canonical representation of the line passing through p1 and p2.
    p1, p2 are tuples (x, y).
    The line is defined by ax + by + c = 0.
    We normalize (a, b, c) such that:
    1. gcd(|a|, |b|, |c|) = 1
    2. a > 0, or a == 0 and b > 0, or a == 0 and b == 0 and c > 0
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    
    # Handle vertical/horizontal/special cases implicitly via gcd
    # If a and b are both 0, it's not a line, but points are distinct so this won't happen for distinct pairs.
    
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    
    # Normalization for sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def _score(points):
    """
    Calculate the score for a set of 16 points.
    Maximize lines with exactly 4 points.
    Penalize lines with >= 5 points.
    """
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            key = _canonical_line(points[i], points[j])
            line_points[key].add(i)
            line_points[key].add(j)
            
    score = 0
    for pts in line_points.values():
        if len(pts) == 4:
            score += 10
        elif len(pts) >= 5:
            score -= 100
            
    return score

def _incremental_score(points, line_points, idx, old_pt, new_pt):
    """
    Update line_points and return new score.
    This is a helper for efficient local search if needed, but for now 
    we will just recompute _score for simplicity given the small N, 
    unless we optimize the loop structure.
    Actually, let's stick to full recompute but optimize the inner loop of SA.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initial random configuration
    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]
        
    points = []
    while len(points) < 16:
        p = random_point()
        if p not in points:
            points.append(p)
            
    best_points = [p[:] for p in points]
    best_score = _score(best_points)
    
    # Simulated Annealing with multiple restarts
    # We use a fixed number of iterations per restart to ensure consistent behavior
    # and avoid infinite loops or uneven time distribution.
    
    num_restarts = 10
    iters_per_restart = 5000
    
    for r in range(num_restarts):
        # New random start for each restart (except the first one which is already set)
        if r > 0:
            points = []
            while len(points) < 16:
                p = random_point()
                if p not in points:
                    points.append(p)
        
        # Local search / SA on this instance
        T = 50.0
        min_T = 0.5
        cooling = 0.999 # Slower cooling for more exploration per restart
        
        for i in range(iters_per_restart):
            if time.time() > deadline:
                return best_points
            
            # Move a random point
            idx = random.randint(0, 15)
            old_x, old_y = points[idx]
            
            # Strategy: 
            # 1. Small perturbation (local search)
            # 2. Large jump (restart-like within SA)
            
            # Bias towards small steps to refine, but allow jumps to escape local minima
            if random.random() < 0.8:
                # Small step
                step = random.randint(1, 3)
                new_x = old_x + random.randint(-step, step)
                new_y = old_y + random.randint(-step, step)
            else:
                # Larger jump
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = [new_x, new_y]
            
            # Check distinctness
            if any(new_point == p for i, p in enumerate(points) if i != idx):
                continue
                
            # Apply move
            points[idx] = new_point
            
            new_score = _score(points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            elif new_score == best_score:
                # Keep if equal? Or accept to explore?
                # Let's accept with some probability to allow wandering
                if random.random() < 0.5:
                    pass # Accept
                else:
                    points[idx] = [old_x, old_y]
            else:
                # Accept worse solution with probability exp(delta/T)
                delta = new_score - best_score
                # delta is negative
                if random.random() < math.exp(delta / T):
                    # Accept
                    pass
                else:
                    # Reject move
                    points[idx] = [old_x, old_y]
                    
            # Cool down
            T *= cooling
            if T < min_T:
                T = min_T
                
    return best_points
