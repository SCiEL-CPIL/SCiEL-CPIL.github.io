<<<<<<< SEARCH
import time
import random
import math
from itertools import combinations

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Normalize
    g = math.gcd(abs(a), abs(b))
    if g == 0:
        g = abs(c)
    if g != 0:
        a //= g
        b //= g
        c //= g
    else:
        return None # Same point, shouldn't happen with distinct points
        
    # Canonical sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
        
    return (a, b, c)

def count_lines(points):
    # points is a list of [x, y]
    n = len(points)
    if n < 4:
        return 0, True
        
    lines = {}
    valid = True
    
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    score = 0
    for key, pts in lines.items():
        cnt = len(pts)
        if cnt >= 5:
            valid = False
            break
        if cnt == 4:
            score += 1
            
    return score, valid

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # Initial configuration: 4x4 grid
    # Center it to allow moves in all directions
    init_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
    
    best_score = -1
    best_points = None
    
    # Evaluate initial
    s, v = count_lines(init_points)
    if v:
        best_score = s
        best_points = [p[:] for p in init_points]
        
    # Local Search
    current_points = [p[:] for p in init_points]
    current_score = best_score
    
    # Define move space
    # Try moving a point to a nearby integer
    # Radius 1-5
    
    while time.time() < deadline:
        improved = False
        # Try random moves
        for _ in range(100): # Iterations per phase
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Generate a new random position nearby
            # Sometimes jump far
            if random.random() < 0.1:
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            else:
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Ensure distinct
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
                
            # Try move
            current_points[idx] = [new_x, new_y]
            
            s, v = count_lines(current_points)
            
            if v and s > current_score:
                current_score = s
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                
        if not improved:
            # Restart from best or random
            if best_points:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
                # Ensure distinct
                while len(set(tuple(p) for p in current_points)) < 16:
                    current_points[random.randint(0,15)] = [random.randint(-10, 10), random.randint(-10, 10)]
                s, v = count_lines(current_points)
                if v:
                    current_score = s
                else:
                    current_score = 0
            
            # If stuck for too long, break? No, just keep trying restarts.
            
    if best_points is None:
        return [[0,0]] * 16 # Fallback, though invalid
        
    return best_points
=======
import time
import random
import math
from collections import defaultdict

def _line_score(points):
    """Fast line counting: returns (score, valid) where score = number of 4-point lines,
    valid = no 5+ collinear."""
    n = len(points)
    if n < 4:
        return 0, True
    
    # Use a dict mapping line -> set of point indices
    lines = defaultdict(set)
    
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            
            # Compute line coefficients (a, b, c) for ax + by + c = 0
            a = dy
            b = -dx
            c = xi * yj - xj * yi
            
            # Normalize by gcd
            g = math.gcd(abs(a), abs(b))
            if g == 0:
                g = abs(c)
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue
            
            # Canonical sign: make first nonzero coefficient positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
    
    score = 0
    valid = True
    for pts in lines.values():
        cnt = len(pts)
        if cnt >= 5:
            valid = False
            break
        if cnt == 4:
            score += 1
    
    return score, valid


def _make_grid_subset(rows, cols, offset_x=0, offset_y=0, seed_offset=0):
    """Create a grid of points, then randomly remove points to get 16."""
    random.seed(seed_offset)
    all_pts = []
    for x in range(-cols // 2 + offset_x, cols - cols // 2 + offset_x):
        for y in range(-rows // 2 + offset_y, rows - rows // 2 + offset_y):
            all_pts.append([x, y])
    
    if len(all_pts) <= 16:
        # If grid is exactly or fewer than 16, use it and pad if needed
        while len(all_pts) < 16:
            all_pts.append([random.randint(-50, 50), random.randint(-50, 50)])
        # Ensure distinct
        seen = set()
        unique = []
        for p in all_pts:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        while len(unique) < 16:
            while True:
                p = [random.randint(-50, 50), random.randint(-50, 50)]
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    unique.append(p)
                    break
        return unique[:16]
    
    # Randomly select 16 points from the grid
    random.shuffle(all_pts)
    return all_pts[:16]


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    best_score = -1
    best_points = None
    
    # Multiple initial configurations from denser grids
    initial_configs = []
    
    # 4x4 grid (baseline)
    initial_configs.append([[x, y] for x in range(-1, 3) for y in range(-1, 3)])
    
    # 5x5 grid subsets (measured to work better)
    for seed_off in [42, 123, 456, 789]:
        initial_configs.append(_make_grid_subset(5, 5, seed_offset=seed_off))
    
    # 6x5 grid subsets
    for seed_off in [111, 222, 333]:
        initial_configs.append(_make_grid_subset(6, 5, seed_offset=seed_off))
    
    # Random starts
    for seed_off in [7, 17, 27, 37]:
        random.seed(seed_off)
        pts = []
        seen = set()
        while len(pts) < 16:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                pts.append(p)
        initial_configs.append(pts)
    
    # Evaluate all initial configs
    for init_pts in initial_configs:
        s, v = _line_score(init_pts)
        if v and s > best_score:
            best_score = s
            best_points = [p[:] for p in init_pts]
    
    if best_points is None:
        # Fallback: simple 4x4
        best_points = [[x, y] for x in range(-1, 3) for y in range(-1, 3)]
        best_score, _ = _line_score(best_points)
    
    # Simulated annealing local search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Temperature schedule
    T = 2.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        T = max(T_min, T * cooling_rate)
        
        improved = False
        num_moves = 0
        max_moves = 200
        
        for _ in range(max_moves):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Generate new position
            r = random.random()
            if r < 0.05:
                # Large jump
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
            elif r < 0.3:
                # Medium jump
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
            else:
                # Small step
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            # Ensure distinct
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(current_points) if i != idx):
                continue
            
            # Try move
            current_points[idx] = [new_x, new_y]
            s, v = _line_score(current_points)
            num_moves += 1
            
            if not v:
                # Invalid: revert with high probability
                current_points[idx] = [old_x, old_y]
                continue
            
            delta = s - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta >= 0:
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current_points]
            else:
                # Accept worse move with probability
                prob = math.exp(-delta / max(T, 0.01))
                if random.random() < prob:
                    current_score = s
                else:
                    current_points[idx] = [old_x, old_y]
        
        # If stuck for a while, restart from best
        if not improved and random.random() < 0.3:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = max(T_min, T * 0.9)  # Cool down
    
    if best_points is None:
        return [[0, 0]] * 16
    
    return best_points
>>>>>>> REPLACE
