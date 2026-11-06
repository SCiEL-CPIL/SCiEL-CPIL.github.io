import time
import itertools
import random

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Use a set of lines, each line represented by a frozenset of point indices
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): normalize the line equation
            # Ax + By + C = 0 where A = y2-y1, B = x1-x2, C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            # Normalize: divide by gcd of A, B, C (handle signs)
            # First, make the first non-zero component positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            # Compute gcd
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _generate_candidates():
    """Generate several candidate initial configurations."""
    candidates = []
    
    # 3x4 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts)
    
    # 4x3 grid
    pts = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts)
    
    # 2x6 grid
    pts = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(pts)
    
    # 6x2 grid
    pts = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(pts)
    
    # Points on two parallel lines with 6 each
    pts = [[x, 0] for x in range(6)] + [[x, 1] for x in range(6)]
    candidates.append(pts)
    
    # Points on three parallel lines with 4 each
    pts = [[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)]
    candidates.append(pts)
    
    # 4 points on x-axis, 4 on y-axis, 4 on diagonal
    pts = [[x, 0] for x in range(4)] + [[0, x] for x in range(1, 5)] + [[x, x] for x in range(1, 5)]
    candidates.append(pts)
    
    # Try a configuration with more structure: 3x4 with some shifted
    pts = [[x, y] for x in range(4) for y in range(3)]
    # Shift some points
    candidates.append(pts)
    
    # Random configurations
    for _ in range(5):
        random.seed(random.randint(0, 1000000))
        pts = []
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if [x, y] not in pts:
                pts.append([x, y])
        candidates.append(pts)
    
    return candidates

def _local_search(points, max_iters=5000, timeout=None):
    """Local search: try moving each point to nearby positions."""
    start_time = time.time()
    best = [p[:] for p in points]
    best_score = _count_ordinary_lines(best)
    
    # Search neighborhood: for each point, try moving it to nearby integer positions
    search_radius = 3
    directions = [(dx, dy) for dx in range(-search_radius, search_radius+1) 
                  for dy in range(-search_radius, search_radius+1) 
                  if dx != 0 or dy != 0]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if timeout and (time.time() - start_time) > timeout:
            break
        
        for i in range(len(best)):
            x, y = best[i]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                # Check if this point already exists
                if new_pt in best:
                    continue
                # Try the move
                old_pt = best[i]
                best[i] = new_pt
                score = _count_ordinary_lines(best)
                if score < best_score:
                    best_score = score
                    improved = True
                else:
                    best[i] = old_pt
    
    return best, best_score

def construct():
    deadline = 120.0  # seconds, with safety margin
    start_time = time.time()
    
    candidates = _generate_candidates()
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate all initial candidates
    for pts in candidates:
        if (time.time() - start_time) > (deadline - 30):
            break
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
    
    # Local search from best candidate
    if best_points and (time.time() - start_time) < (deadline - 10):
        remaining = (deadline - 10) - (time.time() - start_time)
        best_points, best_score = _local_search(best_points, max_iters=2000, timeout=remaining)
    
    # Try local search from a few other candidates if time permits
    for pts in candidates:
        if (time.time() - start_time) > (deadline - 5):
            break
        if [p[:] for p in pts] == best_points:
            continue
        remaining = (deadline - 5) - (time.time() - start_time)
        if remaining < 5:
            break
        cand_pts, cand_score = _local_search([p[:] for p in pts], max_iters=1000, timeout=remaining)
        if cand_score < best_score:
            best_score = cand_score
            best_points = cand_pts
    
    return best_points
