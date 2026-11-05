"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def _evaluate(points):
    """
    Count lines with exactly 4 points. Return -1 if any 5+ collinear.
    Optimized for n=16 by checking all 4-point subsets.
    """
    n = len(points)
    if n < 4:
        return 0
        
    # Pre-convert to tuples for faster access if needed, though list is fine
    pts = points
    count = 0
    
    # Iterate over all combinations of 4 points
    for combo in combinations(range(n), 4):
        p1, p2, p3, p4 = pts[combo[0]], pts[combo[1]], pts[combo[2]], pts[combo[3]]
        
        # Check if p1, p2, p3 are collinear
        dx12 = p2[0] - p1[0]
        dy12 = p2[1] - p1[1]
        dx13 = p3[0] - p1[0]
        dy13 = p3[1] - p1[1]
        
        if dx12 * dy13 != dy12 * dx13:
            continue
            
        # Check if p4 is on the line p1-p2
        dx14 = p4[0] - p1[0]
        dy14 = p4[1] - p1[1]
        
        if dx12 * dy14 != dy12 * dx14:
            continue
            
        # All 4 are collinear. Now check if any other point is on this line.
        # Line equation: dy12 * x - dx12 * y = dy12 * p1[0] - dx12 * p1[1]
        # Let C = dy12 * p1[0] - dx12 * p1[1]
        # Condition: dy12 * x - dx12 * y == C
        
        C = dy12 * p1[0] - dx12 * p1[1]
        valid = True
        for i in range(n):
            if i in combo:
                continue
            pi = pts[i]
            # Check if pi is on the line
            if dy12 * pi[0] - dx12 * pi[1] == C:
                valid = False
                break
        
        if valid:
            count += 1
        else:
            # Found 5 collinear points
            return -1
            
    return count

def _random_points():
    """Generate 16 random distinct integer points."""
    pts = set()
    while len(pts) < 16:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        pts.add((x, y))
    return [list(p) for p in pts]

def _perturb(points):
    """Move one random point to a new random location."""
    idx = random.randrange(16)
    new_pts = [p[:] for p in points]
    while True:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
        if all((x, y) != (p[0], p[1]) for p in new_pts if p is not new_pts[idx]):
            break
    new_pts[idx] = [x, y]
    return new_pts

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Baseline: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _evaluate(best_points)
    
    if best_score < 0:
        best_score = 0
        best_points = _random_points()
        best_score = _evaluate(best_points)
        if best_score < 0:
            best_score = 0

    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Local search with simulated annealing / random restarts
    # Since evaluation is fast, we can do many steps
    temp = 10.0
    
    while time.time() < deadline:
        new_pts = _perturb(current_points)
        new_score = _evaluate(new_pts)
        
        if new_score < 0:
            new_score = -1000
            
        if new_score > current_score:
            current_points = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                # Cool down slightly when improving
                temp = max(1.0, temp * 0.9)
        else:
            # Accept worse with probability
            delta = new_score - current_score
            if random.random() < math.exp(delta / temp):
                current_points = new_pts
                current_score = new_score
        
        temp = max(0.5, temp * 0.999)
        
        # Occasional restart from best or random
        if random.random() < 0.01:
            if random.random() < 0.5:
                current_points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_points = _random_points()
                current_score = _evaluate(current_points)
                if current_score < 0:
                    current_score = -1000
            temp = 10.0

    return best_points
# EVOLVE-BLOCK-END
