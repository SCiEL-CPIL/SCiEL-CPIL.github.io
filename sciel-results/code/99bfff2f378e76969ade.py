"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _evaluate(points):
    """
    Returns (is_valid, score).
    is_valid: False if any 5 points are collinear.
    score: Number of lines containing exactly 4 points.
    """
    n = len(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue # Duplicate points, though construct should prevent this
                
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Normalize direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Canonical line key: (dx, dy, C) where C = dy*x - dx*y
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_pair_counts[key] += 1
            
    valid = True
    score = 0
    
    for count in line_pair_counts.values():
        # Determine k from count: k*(k-1)/2 = count
        k = -1
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        
        if k == -1:
            continue
            
        if k >= 5:
            valid = False
            break
        elif k == 4:
            score += 1
            
    if not valid:
        return False, 0
        
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    valid, best_score = _evaluate(best_points)
    
    random.seed(42)
    
    # Simulated Annealing
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        current = [list(p) for p in best_points]
        current_valid, current_score = _evaluate(current)
        
        # Try a random move: pick a point and move it to a random location
        idx = random.randint(0, 15)
        old_x, old_y = current[idx]
        
        # Generate new location
        if random.random() < 0.8:
            # Local move
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
        else:
            # Global move
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check distinctness
        if any(p[0] == new_x and p[1] == new_y for p in current if p is not current[idx]):
            continue
            
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        
        if not cand_valid:
            cand_score = -1 # Penalty for invalid
            
        # Update global best
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
            
        # Update current state for SA
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
        else:
            delta = cand_score - current_score
            if delta < 0:
                prob = pow(2.718281828, delta / temp)
                if random.random() < prob:
                    current = candidate
                    current_score = cand_score
        # If delta == 0, we keep current to avoid random drift without gain, 
        # or we could accept. Keeping current is safer for convergence.
        
        temp *= cooling
        
    return best_points
# EVOLVE-BLOCK-END
