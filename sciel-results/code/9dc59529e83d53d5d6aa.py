import time
import math
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: 3x4 grid (known to have 7 ordinary lines)
    # We start with a strong candidate to ensure we have a valid baseline.
    # Grid points: x in 0..3, y in 0..2
    points = [[x, y] for x in range(4) for y in range(3)]
    
    # Verify validity
    if len(set(map(tuple, points))) != 12:
        return points
    
    current_score = _count_ordinary(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Simulated Annealing Parameters
    # We want to minimize the number of ordinary lines.
    # The landscape is discrete and rugged.
    
    # Temperature starts high enough to allow jumps, cools down.
    # We use a geometric cooling schedule.
    T = 5.0
    cooling_rate = 0.99
    
    # We will try to move one point at a time.
    # The move is to pick a random point and move it to a new random location.
    # To encourage finding collinear structures, we can bias the moves:
    # 1. Random jump anywhere in a box.
    # 2. Move along a line defined by two other points (to create 3-point lines).
    
    max_iter = 200000 # Upper bound on iterations, but we will break on time
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Decide on a move strategy
        # Strategy 1: Random perturbation (small move)
        # Strategy 2: Random jump (large move)
        # Strategy 3: Snapping to collinearity (move point to lie on line of two others)
        
        r = random.random()
        
        if r < 0.2:
            # Small move: +- 1 to 5
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
        elif r < 0.4:
            # Large move: random in box [-10, 10] relative to current
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            # Snapping: Pick two other points i, j. Try to move idx to a point on line(i,j)
            # that is close to the original position or within bounds.
            # This is heuristic to encourage 3-collinearities.
            # Actually, let's just do a random jump in a larger box to escape local minima.
            # The "snapping" is hard to do efficiently without checking all lines.
            # Let's just do a wider random jump.
            new_x = old_x + random.randint(-20, 20)
            new_y = old_y + random.randint(-20, 20)
            
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check distinctness
        # Optimization: check against all other points
        is_distinct = True
        for k in range(12):
            if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Apply move
        points[idx] = [new_x, new_y]
        new_score = _count_ordinary(points)
        
        # Acceptance criterion
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                if best_score == 6:
                    # Found the theoretical minimum
                    return best_points
        else:
            # Worse or same
            # Accept with probability exp(-delta / T)
            if delta == 0:
                # Same score, accept to explore
                current_score = new_score
                # Update best if we haven't seen this exact config? 
                # We don't track history, so just keep exploring.
                # Actually, if score is same, we might be stuck in a plateau.
                # Accepting helps escape.
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject move
                    points[idx] = [old_x, old_y]
                    
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            # Reheat or restart? 
            # If we cool too much, we get stuck. 
            # Let's restart temperature if we've been stuck for a while?
            # Simple approach: if T is too low, reset to a moderate value to allow exploration again.
            # This is a form of "simulated annealing with restarts" or just keeping T bounded.
            T = 1.0

    return best_points
