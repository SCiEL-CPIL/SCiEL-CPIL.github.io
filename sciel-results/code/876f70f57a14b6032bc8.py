"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _evaluate(points):
    """Count lines with exactly 4 points. Return -1 if any line has 5+ points."""
    n = len(points)
    # Map each line (represented by canonical form) to the set of points on it
    # Use (dx, dy, x0, y0) where dx,dy are reduced and (x0,y0) is a reference
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_r = dx // g
            dy_r = dy // g
            # Canonical direction: make dx positive, or if dx==0 make dy positive
            if dx_r < 0 or (dx_r == 0 and dy_r < 0):
                dx_r = -dx_r
                dy_r = -dy_r
            # Find the "base" point: the point on this line with smallest x, then smallest y
            # Actually, use a canonical representation: the line is determined by (dx_r, dy_r) and the intercept
            # Use: dy_r * x - dx_r * y = c, where c is normalized
            c = dy_r * x1 - dx_r * y1
            key = (dx_r, dy_r, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    four_count = 0
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return -1
        if len(pts) == 4:
            four_count += 1
    return four_count

def _random_point(exclude, bound=50):
    """Generate a random point not in exclude set."""
    while True:
        x = random.randint(-bound, bound)
        y = random.randint(-bound, bound)
        p = (x, y)
        if p not in exclude:
            return p

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Initial: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _evaluate(best_points)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9999
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_pt = current_points[idx]
        
        # Generate new position: random perturbation
        # Sometimes small move, sometimes larger
        if random.random() < 0.7:
            new_pt = [old_pt[0] + random.randint(-3, 3), old_pt[1] + random.randint(-3, 3)]
        else:
            new_pt = [random.randint(-50, 50), random.randint(-50, 50)]
        
        # Check bounds
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            new_pt = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        
        # Check distinctness
        new_pt = tuple(new_pt)
        if any(tuple(p) == new_pt for i, p in enumerate(current_points) if i != idx):
            continue
        
        # Try the move
        current_points[idx] = list(new_pt)
        new_score = _evaluate(current_points)
        
        if new_score >= 0:
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            elif new_score == current_score:
                pass  # accept neutral
            else:
                # Accept with probability exp((new_score - current_score) / T)
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    pass  # accept
                else:
                    current_points[idx] = list(old_pt)
        else:
            # Invalid: reject
            current_points[idx] = list(old_pt)
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally restart with perturbation if stuck
        if iteration % 5000 == 0 and best_score <= 10:
            # Try a random restart with some structure
            if random.random() < 0.3:
                # Perturb the best solution
                current_points = [p[:] for p in best_points]
                current_score = best_score
                for _ in range(random.randint(1, 3)):
                    idx2 = random.randint(0, 15)
                    current_points[idx2] = [random.randint(-50, 50), random.randint(-50, 50)]
                current_score = _evaluate(current_points)
                if current_score < 0:
                    current_points = [p[:] for p in best_points]
                    current_score = best_score
    
    # Final check
    result = _evaluate(best_points)
    if result < 0:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
