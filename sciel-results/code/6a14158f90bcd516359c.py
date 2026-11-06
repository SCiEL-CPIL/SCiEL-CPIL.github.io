import time
import random
import math
from itertools import combinations

def _count_four_point_lines(points):
    """Count lines containing exactly 4 of the points."""
    n = len(points)
    count = 0
    seen_lines = set()
    # To speed up, we can iterate over pairs and find lines, but N=16 is small.
    # However, checking all 4-combinations is O(N^4) = 43680 checks.
    # Each check is O(1) or O(N) for the 5th point check.
    # Total O(N^5) ~ 1M ops. Very fast.
    
    # Optimization: Precompute line keys for all pairs?
    # No, just use the combination method but optimize the inner loop.
    
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize direction
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                dxn, dyn = 0, 0 # Should not happen for distinct points
            
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
                
            # Line equation: a*x + b*y = c
            # Normal vector (a, b) is perpendicular to (dxn, dyn)
            # So a = dyn, b = -dxn
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            
            line_key = (a, b, c)
            
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            # Count points on this line
            pts_on_line = []
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    pts_on_line.append(k)
                    
            if len(pts_on_line) == 4:
                count += 1
            elif len(pts_on_line) >= 5:
                # Invalid configuration for the objective "exactly 4", 
                # but the problem says "no five collinear". 
                # So if we find 5, the config is invalid.
                # We should return -1 or a penalty.
                return -1
    return count

def _is_valid(points):
    """Check that no five points are collinear."""
    n = len(points)
    # Check distinctness
    seen = set()
    for p in points:
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
        
    # Check no 5 collinear
    # We can use the line counting logic
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                continue
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            cnt = 0
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    cnt += 1
            if cnt >= 5:
                return False
    return True

def _evaluate(points):
    """Returns score, or -1 if invalid."""
    if not _is_valid(points):
        return -1
    return _count_four_point_lines(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Define a grid of candidate points
    # 6x6 grid centered at 0
    grid_points = []
    for x in range(-2, 4):
        for y in range(-2, 4):
            grid_points.append([x, y])
            
    # We need 16 points. Grid has 36.
    # Random initialization
    best_points = None
    best_score = -1
    
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        # Select 16 random points from the grid
        current_points = [p[:] for p in random.sample(grid_points, 16)]
        
        # Initial evaluation
        current_score = _evaluate(current_points)
        if current_score < 0:
             # Try to fix or just restart
             continue
             
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        while T > T_min and time.time() < deadline - 5:
            # Pick a point to move
            i = random.randint(0, 15)
            old_point = current_points[i][:]
            
            # Pick a new point from the grid that is not already in the set
            # To make this efficient, we can just pick a random grid point and check
            # or maintain a list of available points.
            # For simplicity, pick random from grid, if duplicate, try again.
            new_point = None
            for _ in range(100):
                candidate = random.choice(grid_points)
                if candidate not in current_points:
                    new_point = candidate
                    break
            
            if new_point is None:
                continue
                
            current_points[i] = new_point[:]
            
            new_score = _evaluate(current_points)
            
            if new_score < 0:
                # Invalid
                delta = -100
                accept = False
            else:
                delta = new_score - current_score
                if delta > 0:
                    accept = True
                else:
                    # Accept with probability exp(delta / T)
                    if random.random() < math.exp(delta / T):
                        accept = True
                    else:
                        accept = False
            
            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i] = old_point
            
            T *= cooling
            
    # Final check
    if best_points is None:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
