import time
import sys
from math import gcd
from itertools import combinations
from collections import defaultdict

def _count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            g2 = gcd(abs(a), abs(b))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            line_map[(a, b, c)].add(i)
            line_map[(a, b, c)].add(j)
    
    ordinary = 0
    for line, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def _try_config(points):
    """Validate and count ordinary lines for a configuration."""
    if len(points) != 12:
        return None
    if len(set(tuple(p) for p in points)) != 12:
        return None
    for p in points:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return None
    collinear = True
    for i in range(1, 12):
        dx = points[i][0] - points[0][0]
        dy = points[i][1] - points[0][1]
        for j in range(i + 1, 12):
            dx2 = points[j][0] - points[0][0]
            dy2 = points[j][1] - points[0][1]
            if dx * dy2 != dy * dx2:
                collinear = False
                break
        if not collinear:
            break
    if collinear:
        return None
    return _count_ordinary_lines(points)

def _local_search(points, max_iters=50, time_limit=None, start_time=None):
    """Local search: try moving each point to reduce ordinary lines."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 120
    
    current = [p[:] for p in points]
    current_count = _count_ordinary_lines(current)
    best = current[:]
    best_count = current_count
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for idx in range(12):
            if time.time() - start_time > time_limit:
                break
            
            orig_x, orig_y = current[idx]
            # Try moving this point to various positions
            # Strategy: try to align with existing lines or create new 3+ point lines
            other_points = [current[j] for j in range(12) if j != idx]
            
            # Generate candidate positions:
            # 1. Points that would create a 3-point line with two existing points
            candidates = set()
            for i, j in combinations(range(len(other_points)), 2):
                p1 = other_points[i]
                p2 = other_points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Extend the line beyond p1 or p2
                for sign in [-1, 1]:
                    for mult in [1, 2]:
                        nx = p1[0] + sign * mult * dx
                        ny = p1[1] + sign * mult * dy
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.add((nx, ny))
                        nx = p2[0] + sign * mult * dx
                        ny = p2[1] + sign * mult * dy
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            candidates.add((nx, ny))
            
            # Also try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx = orig_x + dx
                    ny = orig_y + dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        candidates.add((nx, ny))
            
            # Limit candidates to keep runtime manageable
            if len(candidates) > 200:
                # Prioritize candidates
                candidates = list(candidates)[:200]
            
            for (nx, ny) in candidates:
                if time.time() - start_time > time_limit:
                    break
                if (nx, ny) == (orig_x, orig_y):
                    continue
                # Check if this position is already taken
                if any(p[0] == nx and p[1] == ny for p in other_points):
                    continue
                
                new_points = [p[:] for p in current]
                new_points[idx] = [nx, ny]
                new_count = _count_ordinary_lines(new_points)
                
                if new_count < current_count:
                    current = new_points
                    current_count = new_count
                    if current_count < best_count:
                        best = current[:]
                        best_count = current_count
                    improved = True
                    break
        
        if not improved:
            break
    
    return best, best_count

def construct():
    """
    Constructs 12 distinct integer points minimizing ordinary lines.
    Tests several known configurations and applies local search.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Candidate configurations
    candidates = []
    
    # Config 1: 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # Config 2: 3x4 grid
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # Config 3: 11 on line + 1 off
    pts3 = [[i, 0] for i in range(11)] + [[0, 1]]
    candidates.append(pts3)
    
    # Config 4: 6 on x-axis, 6 on y-axis (cross)
    pts4 = [[i, 0] for i in range(6)] + [[0, i] for i in range(1, 7)]
    candidates.append(pts4)
    
    # Config 5: Points on two parallel lines (6 each)
    pts5 = [[i, 0] for i in range(6)] + [[i, 1] for i in range(6)]
    candidates.append(pts5)
    
    # Config 6: 4x3 grid shifted
    pts6 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts6)
    
    # Config 7: 2x6 grid
    pts7 = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(pts7)
    
    # Config 8: Hexagonal-ish arrangement
    pts8 = [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], [0,2], [1,2], [2,2], [3,2]]
    candidates.append(pts8)
    
    # Config 9: 3 lines of 4 points (non-concurrent)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=1,2,3,4
    # Line 3: x=0, y=0,1,2,3 -> shares (0,0) and (0,1) with above
    # Let me do: 
    # L1: (0,0),(1,0),(2,0),(3,0)
    # L2: (1,1),(2,1),(3,1),(4,1)
    # L3: (0,0),(0,1),(0,2),(0,3) -> shares (0,0)
    # Unique: (0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(4,1),(0,1),(0,2),(0,3) = 11 points
    # Add one more: (1,2)
    pts9 = [[0,0],[1,0],[2,0],[3,0],[1,1],[2,1],[3,1],[4,1],[0,1],[0,2],[0,3],[1,2]]
    candidates.append(pts9)
    
    # Config 10: 4 concurrent lines through origin
    # y=0: (1,0),(2,0),(3,0)
    # y=x: (1,1),(2,2),(3,3)
    # y=2x: (1,2),(2,4),(3,6)
    # y=-x: (1,-1),(2,-2),(3,-3)
    # Plus origin: total 1+12=13, too many
    # Use 3 lines: 1 + 3*3 = 10, add 2 more
    pts10 = [[0,0]]
    for i in range(1, 4):
        pts10.append([i, 0])
        pts10.append([i, i])
        pts10.append([i, 2*i])
    # 10 points, add 2 more
    pts10.append([0, 1])
    pts10.append([1, -1])
    candidates.append(pts10)
    
    # Find the best configuration with local search
    best_points = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # First get the static count
        static_count = _try_config(pts)
        if static_count is not None and static_count < best_count:
            best_count = static_count
            best_points = [p[:] for p in pts]
        
        # Apply local search
        if time.time() > deadline - 30:  # Reserve time for remaining candidates
            continue
            
        searched, search_count = _local_search(pts, max_iters=30, time_limit=deadline - start_time, start_time=start_time)
        if search_count < best_count:
            best_count = search_count
            best_points = searched
    
    # If no valid config found, fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
