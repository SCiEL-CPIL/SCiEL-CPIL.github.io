"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

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
            key = _line_key(x1, y1, points[j][0], points[j][1])
            if key is not None:
                line_pair_counts[key] += 1
            
    valid = True
    score = 0
    
    for count in line_pair_counts.values():
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

def _local_search(points, deadline):
    """Try relocating each point to improve score."""
    current = [list(p) for p in points]
    valid, score = _evaluate(current)
    if not valid:
        return current, score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(16):
            if time.time() >= deadline:
                break
            old_x, old_y = current[idx]
            # Try nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check distinctness
                    if any(p[0] == nx and p[1] == ny for i, p in enumerate(current) if i != idx):
                        continue
                    cand = [list(p) for p in current]
                    cand[idx] = [nx, ny]
                    cvalid, cscore = _evaluate(cand)
                    if cvalid and cscore > score:
                        current = cand
                        score = cscore
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return current, score

def _generate_initial_5x5():
    """Generate a good initial 5x5 grid subset with 16 points."""
    # Start with full 5x5 grid (25 points) and remove 9
    # Try to keep points that form many 4-point lines
    all_points = [[x, y] for x in range(5) for y in range(5)]
    
    # We need to pick 16 from 25
    # Use a heuristic: start with all, remove points that are "least useful"
    # A point is useful if it's on many lines with other points
    
    best_subset = None
    best_score = -1
    
    # Try several random removals
    random.seed(123)
    for trial in range(50):
        subset = list(all_points)
        random.shuffle(subset)
        subset = subset[:16]
        valid, score = _evaluate(subset)
        if valid and score > best_score:
            best_score = score
            best_subset = [list(p) for p in subset]
    
    return best_subset, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial configuration: 5x5 grid subset (better than 4x4)
    best_points, best_score = _generate_initial_5x5()
    
    # Also try 4x4 grid as fallback
    grid4 = [[x, y] for x in range(4) for y in range(4)]
    v4, s4 = _evaluate(grid4)
    if v4 and s4 > best_score:
        best_points = grid4
        best_score = s4
    
    random.seed(42)
    
    # Simulated Annealing with structured moves
    temp = 5.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        current = [list(p) for p in best_points]
        current_valid, current_score = _evaluate(current)
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Local jitter: move one point slightly
            idx = random.randint(0, 15)
            old_x, old_y = current[idx]
            new_x = old_x + random.randint(-8, 8)
            new_y = old_y + random.randint(-8, 8)
        elif move_type < 0.8:
            # Try to place on an existing line
            # Pick a random pair of points, find line, try to place third point on it
            i1 = random.randint(0, 15)
            i2 = random.randint(0, 15)
            if i1 == i2:
                continue
            x1, y1 = current[i1]
            x2, y2 = current[i2]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Generate candidate points on this line
            t_vals = random.sample(range(-50, 51), 5)
            found = False
            for t in t_vals:
                nx = x1 + dx * t
                ny = y1 + dy * t
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if any(p[0] == nx and p[1] == ny for i, p in enumerate(current) if i not in (i1, i2)):
                    continue
                # Move a random third point to this location
                idx = random.randint(0, 15)
                if idx in (i1, i2):
                    continue
                new_x, new_y = nx, ny
                found = True
                break
            if not found:
                continue
        else:
            # Global random move
            idx = random.randint(0, 15)
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            no_improve_count += 1
            continue
            
        # Check distinctness
        if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current) if i != idx):
            no_improve_count += 1
            continue
            
        candidate = [list(p) for p in current]
        candidate[idx] = [new_x, new_y]
        
        cand_valid, cand_score = _evaluate(candidate)
        
        if not cand_valid:
            cand_score = -1
            
        # Update global best
        if cand_score > best_score:
            best_points = candidate
            best_score = cand_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Update current state for SA
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
        else:
            delta = cand_score - current_score
            if delta < 0:
                prob = pow(2.718281828, delta / max(temp, 0.01))
                if random.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        temp *= cooling
    
    # Final local search polish
    if time.time() < deadline - 5:
        polished, polished_score = _local_search(best_points, deadline - 2)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points
# EVOLVE-BLOCK-END
