import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            if A == 0 and B == 0:
                continue
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
            elif A == 0 and B == 0 and C == 0:
                A = 1
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def _is_valid(points):
    """Check validity: distinct, integer, bounds, not all collinear."""
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check not all collinear
    p0 = points[0]
    p1 = points[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    for i in range(2, 12):
        pi = points[i]
        if dx * (pi[1] - p0[1]) != dy * (pi[0] - p0[0]):
            return True
            
    return False

def _simulate_annealing(initial_points, max_time=10.0):
    """
    Run Simulated Annealing to minimize ordinary lines.
    """
    start_time = time.time()
    current_points = [list(p) for p in initial_points]
    current_score = _count_ordinary_lines(current_points)
    
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    temperature = 5.0
    cooling = 0.99
    
    while time.time() - start_time < max_time and temperature > 0.1:
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = current_points[i]
        
        # Generate a new random position in a local neighborhood
        # Use a box of size 5 around current
        radius = 5
        new_x = old_x + random.randint(-radius, radius)
        new_y = old_y + random.randint(-radius, radius)
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check collision
        collision = False
        for j in range(12):
            if j == i:
                continue
            if current_points[j][0] == new_x and current_points[j][1] == new_y:
                collision = True
                break
        if collision:
            continue
            
        # Apply move
        current_points[i] = [new_x, new_y]
        
        # Check validity (not all collinear)
        # Fast check: if score is 0, it might be all collinear.
        # But usually we just check if the move is valid.
        # We can skip full _is_valid for speed, but must ensure not all collinear.
        # If current_score > 0, it's not all collinear.
        # If current_score becomes 0, we must check.
        
        new_score = _count_ordinary_lines(current_points)
        
        if new_score == 0:
             # Check if all collinear
             is_all_collinear = True
             p0 = current_points[0]
             p1 = current_points[1]
             dx = p1[0] - p0[0]
             dy = p1[1] - p0[1]
             for k in range(2, 12):
                 pk = current_points[k]
                 if dx * (pk[1] - p0[1]) != dy * (pk[0] - p0[0]):
                     is_all_collinear = False
                     break
             if is_all_collinear:
                 current_points[i] = [old_x, old_y]
                 continue

        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < pow(2.718, -delta / temperature):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [list(p) for p in current_points]
                if best_score == 6:
                    return best_points, best_score
        else:
            # Revert
            current_points[i] = [old_x, old_y]
            
        temperature *= cooling
        
    return best_points, best_score

def construct():
    """
    Construct 12 integer points with minimized ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    # 1. 4x3 Grid
    grid_4x3 = [[i, j] for i in range(4) for j in range(3)]
    
    # 2. 3x4 Grid
    grid_3x4 = [[i, j] for i in range(3) for j in range(4)]
    
    # 3. Points on parabola y=x^2
    parabola = [[x, x*x] for x in range(12)]
    
    # 4. Random small integers
    random.seed(42)
    random_config = []
    used = set()
    while len(random_config) < 12:
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        if (x, y) not in used:
            used.add((x, y))
            random_config.append([x, y])
            
    initial_configs = [grid_4x3, grid_3x4, parabola, random_config]
    
    # Add more random restarts
    for _ in range(10):
        rc = []
        used = set()
        while len(rc) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used:
                used.add((x, y))
                rc.append([x, y])
        initial_configs.append(rc)
        
    # Run SA on each config
    for config in initial_configs:
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        time_per_config = min(10.0, remaining / 2.0) # Leave time for remaining configs
        
        if time_per_config < 1.0:
            continue
            
        pts, score = _simulate_annealing(config, max_time=time_per_config)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score == 6:
            break
            
    # Final validation
    if best_points is None:
        best_points = grid_4x3
        
    if not _is_valid(best_points):
        best_points = grid_4x3
        
    return best_points
