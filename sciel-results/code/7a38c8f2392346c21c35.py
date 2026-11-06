"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return canonical line key for two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    if a == 0 and b == 0:
        return None
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a, b, c = a // g, b // g, c // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary_fast(points):
    """Count ordinary lines using a dict of line->count."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for v in line_counts.values() if v == 1)

def _is_valid(points):
    """Check: 12 distinct points, not all collinear."""
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Check not all collinear
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[1], points[2]
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return True
    for i in range(3, 12):
        (x4, y4) = points[i]
        if (x2 - x1) * (y4 - y1) != (y2 - y1) * (x4 - x1):
            return True
    return False

def _local_search(points, deadline):
    """Try single-point replacement moves to reduce ordinary line count."""
    current = [list(p) for p in points]
    current_score = _count_ordinary_fast(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Candidate pool: points near the current configuration
    coords = set()
    for p in current:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                coords.add((p[0] + dx, p[1] + dy))
    # Also add some structured candidates
    for x in range(-1, 5):
        for y in range(-1, 5):
            coords.add((x, y))
    
    candidate_list = list(coords)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        random.shuffle(candidate_list)
        
        for i in range(12):
            if time.time() >= deadline:
                break
            old_pt = current[i]
            for cand in candidate_list:
                if tuple(cand) == tuple(old_pt):
                    continue
                if any(tuple(cand) == tuple(current[j]) for j in range(12) if j != i):
                    continue
                
                new_pts = [list(p) for p in current]
                new_pts[i] = list(cand)
                
                if not _is_valid(new_pts):
                    continue
                
                score = _count_ordinary_fast(new_pts)
                if score < current_score:
                    current = new_pts
                    current_score = score
                    improved = True
                    no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
                    break
            else:
                no_improve += 1
                continue
            break
        else:
            no_improve += 1
            continue
        
        if not improved:
            no_improve += 1
    
    return best, best_score

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 135  # Safety margin
    
    # Start with 3x4 grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    best_points = grid_3x4
    best_score = _count_ordinary_fast(grid_3x4)
    
    # Try local search from grid
    if time.time() < deadline - 5:
        ls_points, ls_score = _local_search(grid_3x4, deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
    
    # If we haven't reached 6, try other starting configurations
    if best_score > 6 and time.time() < deadline - 5:
        # Try 2x6 grid
        grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
        s2 = _count_ordinary_fast(grid_2x6)
        if s2 < best_score:
            best_score = s2
            best_points = grid_2x6
        if time.time() < deadline - 5:
            ls_points, ls_score = _local_search(grid_2x6, deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points
    
    # Try a few more structured starts
    if best_score > 6 and time.time() < deadline - 5:
        # Points on a conic-like arrangement
        # 4 points on y=0, 4 on y=1, 2 on y=2, 2 on y=3
        start3 = [[0,0],[1,0],[2,0],[3,0],
                  [0,1],[1,1],[2,1],[3,1],
                  [0,2],[1,2],
                  [0,3],[1,3]]
        s3 = _count_ordinary_fast(start3)
        if s3 < best_score:
            best_score = s3
            best_points = start3
        if time.time() < deadline - 5:
            ls_points, ls_score = _local_search(start3, deadline)
            if ls_score < best_score:
                best_score = ls_score
                best_points = ls_points
    
    # Random restarts if still above 6
    if best_score > 6 and time.time() < deadline - 10:
        random.seed(42)
        for restart in range(20):
            if time.time() >= deadline - 5:
                break
            # Generate a random-ish configuration with some structure
            pts = []
            # Place 4 points on each of 3 lines
            # Line 1: y = 0
            for x in range(4):
                pts.append([x, 0])
            # Line 2: y = x + 1 (shifted to avoid overlap)
            for x in range(4):
                pts.append([x + 1, x + 1])
            # Line 3: y = 2x + 2
            for x in range(4):
                pts.append([x + 2, 2*x + 2])
            
            if not _is_valid(pts):
                continue
            
            s = _count_ordinary_fast(pts)
            if s < best_score:
                best_score = s
                best_points = pts
            
            # Local search from here
            if time.time() < deadline - 5:
                ls_points, ls_score = _local_search(pts, deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_points = ls_points
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
