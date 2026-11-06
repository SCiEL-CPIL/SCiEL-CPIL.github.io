"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return normalized line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = _gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _count_lines(points):
    """Count lines with exactly 4 points, and check for 5+ collinear."""
    n = len(points)
    line_to_indices = defaultdict(set)
    
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_to_indices[key].add(i)
                line_to_indices[key].add(j)
    
    exact_4 = 0
    has_5_plus = False
    for key, idx_set in line_to_indices.items():
        cnt = len(idx_set)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5_plus = True
    
    return exact_4, has_5_plus

def _score(points):
    """Return score: number of exact-4 lines if no 5+ collinear, else -1."""
    exact_4, has_5_plus = _count_lines(points)
    if has_5_plus:
        return -1
    return exact_4

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    random.seed(42)
    
    # Try multiple restarts with different initial configurations
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            current_points = [list(p) for p in best_points]
        else:
            # Generate a random initial configuration
            current_points = []
            used = set()
            while len(current_points) < 16:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    used.add((x, y))
                    current_points.append([x, y])
        
        current_score = _score(current_points)
        if current_score > best_score:
            best_score = current_score
            best_points = [list(p) for p in current_points]
        
        # Local search
        T = 5.0
        cooling = 0.99995
        no_improve = 0
        
        while time.time() < deadline - 5:
            # Choose a random point to move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Try a random new position
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            # Check if this creates a duplicate
            is_dup = False
            for j in range(16):
                if j != idx and current_points[j][0] == new_x and current_points[j][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = _score(current_points)
            
            if new_score >= 0:
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [list(p) for p in current_points]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    delta = new_score - current_score
                    if delta < 0:
                        accept_prob = math.exp(delta / T) if delta / T > -50 else 0.0
                        if random.random() < accept_prob:
                            no_improve += 1
                        else:
                            current_points[idx] = [old_x, old_y]
                            no_improve += 1
                    else:
                        no_improve += 1
            else:
                current_points[idx] = [old_x, old_y]
                no_improve += 1
            
            T *= cooling
            
            if no_improve > 500:
                break
        
        # If we found something better, continue; otherwise restart
        if best_score > 10:
            # Try to refine further
            pass
    
    return best_points
# EVOLVE-BLOCK-END
