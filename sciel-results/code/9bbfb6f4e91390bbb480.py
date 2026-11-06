"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _get_line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score_and_check(points):
    """
    Returns (score, is_valid).
    score: number of lines with exactly 4 points.
    is_valid: True if no 5 collinear.
    """
    n = len(points)
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            key = _get_line_key(x1, y1, x2, y2)
            if key:
                line_counts[key] += 1
    
    score = 0
    is_valid = True
    for cnt in line_counts.values():
        # C(k, 2) pairs.
        # 4 points -> 6 pairs
        # 5 points -> 10 pairs
        if cnt == 6:
            score += 1
        elif cnt >= 10:
            is_valid = False
    
    # Check bounds and distinctness (fast check)
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
        if (x, y) in seen:
            return 0, False
        seen.add((x, y))
        
    if not is_valid:
        return 0, False
        
    return score, True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s safety margin from 150s limit
    
    # Start with 4x4 grid (shifted to center 0,0 to allow movement)
    best_points = [[x - 2, y - 2] for x in range(4) for y in range(4)]
    best_score, _ = _score_and_check(best_points)
    
    random.seed(42)
    
    # To avoid infinite loop of failing restarts, we limit total steps or use a counter
    steps = 0
    max_steps = 200000 # Hard cap on evaluations to ensure termination
    
    while time.time() < deadline and steps < max_steps:
        improved = False
        # Local search: try moving individual points
        # Shuffle indices to avoid bias
        indices = list(range(16))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline or steps >= max_steps:
                break
            orig_x, orig_y = best_points[i]
            # Try nearby positions
            # Use a smaller radius for speed, or random jumps
            # Let's try random jumps in a small box
            for _ in range(20): # 20 random attempts per point
                if time.time() > deadline or steps >= max_steps:
                    break
                new_x = orig_x + random.randint(-5, 5)
                new_y = orig_y + random.randint(-5, 5)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                # Check if position is taken
                new_pos = (new_x, new_y)
                if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(best_points) if j != i):
                    continue
                
                # Try the move
                best_points[i] = [new_x, new_y]
                s, valid = _score_and_check(best_points)
                steps += 1
                
                if valid and s > best_score:
                    best_score = s
                    improved = True
                else:
                    best_points[i] = [orig_x, orig_y]
        
        if not improved:
            # Try random restarts with larger perturbations
            for _ in range(20):
                if time.time() > deadline or steps >= max_steps:
                    break
                candidate = [p[:] for p in best_points]
                # Perturb 2-3 points more aggressively
                for _ in range(random.randint(2, 3)):
                    idx = random.randint(0, 15)
                    candidate[idx][0] += random.randint(-10, 10)
                    candidate[idx][1] += random.randint(-10, 10)
                
                s, valid = _score_and_check(candidate)
                steps += 1
                
                if valid and s > best_score:
                    best_score = s
                    best_points = candidate
                    improved = True
    
    return [p[:] for p in best_points]
# EVOLVE-BLOCK-END
