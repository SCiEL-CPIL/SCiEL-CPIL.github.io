"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import itertools
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    Optimized for N=12.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map line (defined by normalized equation) to count of points on it
    # Instead of storing sets, we can just count how many times a line is formed by pairs?
    # No, that overcounts.
    # Better: For each point, count slopes to other points.
    # If a point i has k other points on the same line through it, that line has k+1 points.
    # We need to identify unique lines.
    
    # Standard O(N^2) approach:
    # For each point i, compute slopes to all other points j.
    # Group by slope. If a group has size k, there is a line with k+1 points.
    # But this counts lines multiple times (once for each point on the line).
    # A line with m points is counted m times in this way.
    # We want to count lines with exactly 2 points.
    
    # Let's stick to the dictionary of lines but make it faster.
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add((x1, y1))
            line_points[key].add((x2, y2))
            
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _generate_initial_points():
    # Generate a few diverse starting configurations
    configs = []
    
    # 1. The "star" configuration from the original code
    p1 = [
        [0, 0],
        [1, 0], [2, 0], [3, 0],
        [0, 1], [0, 2], [0, 3],
        [1, 1], [2, 2], [3, 3],
        [1, 2], [2, 1]
    ]
    if len(set(map(tuple, p1))) == 12:
        configs.append(p1)
        
    # 2. 3x4 Grid
    p2 = [[x, y] for x in range(4) for y in range(3)]
    configs.append(p2)
    
    # 3. Random points in a small box
    random.seed(42)
    for _ in range(5):
        pts = []
        seen = set()
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        configs.append(pts)
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    initial_configs = _generate_initial_points()
    
    best_points = None
    best_score = float('inf')
    
    # Evaluate initial configs
    for pts in initial_configs:
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
    # Local Search with Smart Moves
    # Strategy:
    # 1. Pick a point to move.
    # 2. Generate candidate moves:
    #    a. Random nearby points.
    #    b. Points that lie on the line defined by two other points (to create 3-collinear).
    #    c. Points that break an ordinary line (move off a line with only 2 points).
    
    max_restarts = 20
    max_iters = 1000
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_points = [p[:] for p in best_points]
            current_score = best_score
        else:
            # Restart from a perturbed version of the best
            current_points = []
            for p in best_points:
                if random.random() < 0.3:
                    current_points.append([p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5)])
                else:
                    current_points.append(p[:])
            
            # Ensure distinctness
            seen = set()
            valid = True
            for p in current_points:
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            if not valid:
                continue
                
            current_score = _count_ordinary_lines(current_points)
        
        no_improve_count = 0
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a point to move
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Generate candidate moves
            candidates = set()
            
            # 1. Random nearby
            for _ in range(5):
                nx = old_x + random.randint(-3, 3)
                ny = old_y + random.randint(-3, 3)
                candidates.add((nx, ny))
                
            # 2. Smart move: Create a new collinearity
            # Pick two other points i, j. Calculate line.
            # Find an integer point on this line that is not i, j, or idx.
            # This is complex to generate directly.
            # Alternative: Just try to move idx to a point that is collinear with two others.
            # For each pair (i, j) where i != idx and j != idx:
            #   The line through i and j is fixed.
            #   We want to move idx to a point P on this line.
            #   If we do, the line (i, j, idx) will have 3 points.
            #   This reduces ordinary lines if (i, j) was ordinary or if idx was part of other ordinary lines.
            
            # Let's implement a simpler smart move:
            # Move idx to a point that is the "reflection" or extension of an existing segment.
            
            for i in range(12):
                if i == idx:
                    continue
                for j in range(12):
                    if j == idx or j == i:
                        continue
                    x1, y1 = current_points[i]
                    x2, y2 = current_points[j]
                    # Vector
                    dx = x2 - x1
                    dy = y2 - y1
                    # Try extending from x1 or x2
                    # P = x1 + k * (dx, dy)
                    # We want integer coordinates.
                    # If dx, dy are small, we can try k=2, -1, etc.
                    # To keep it simple and fast, only try if gcd(dx, dy) is small?
                    # Actually, let's just try a few specific positions relative to i and j.
                    
                    # Midpoint? (x1+x2)/2 might not be integer.
                    # Extension: x1 - (x2-x1), y1 - (y2-y1) -> (2*x1 - x2, 2*y1 - y2)
                    nx = 2 * x1 - x2
                    ny = 2 * y1 - y2
                    if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                        candidates.add((nx, ny))
                        
                    nx = 2 * x2 - x1
                    ny = 2 * y2 - y1
                    if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                        candidates.add((nx, ny))
            
            # Filter candidates: must be distinct from all other points
            valid_candidates = []
            existing = set(map(tuple, current_points))
            existing.discard((old_x, old_y)) # idx is moving
            
            for cx, cy in candidates:
                if (cx, cy) not in existing:
                    valid_candidates.append([cx, cy])
            
            if not valid_candidates:
                continue
                
            # Pick a random valid candidate
            new_x, new_y = random.choice(valid_candidates)
            
            # Apply move
            current_points[idx] = [new_x, new_y]
            new_score = _count_ordinary_lines(current_points)
            
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                no_improve_count += 1
                # Revert if no improvement for a while? Or just accept equal?
                # Let's revert to escape local minima slightly
                if no_improve_count > 50:
                    current_points[idx] = [old_x, old_y]
                    no_improve_count = 0
                    # Break out of inner loop to restart?
                    break
            
        # Update best after restart
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
    return best_points
# EVOLVE-BLOCK-END
