import numpy as np
import time
from itertools import combinations
from math import gcd

def get_lines_and_points_for_grid(W, H, removals):
    """
    Generate points for a W x H grid with specified removals, 
    and find the best 27 lines.
    """
    # 1. Generate Points
    points = []
    for y in range(H):
        for x in range(W):
            if (x, y) not in removals:
                points.append((x, y))
    
    points = np.array(points, dtype=np.int64)
    n_points = len(points)
    if n_points != 54:
        return None, None, 0

    # 2. Generate Candidate Lines
    # A line is determined by two points.
    # We store lines as normalized (A, B, C)
    line_counts = {}
    
    # Precompute point indices for fast lookup? 
    # Or just check all points for each candidate line.
    # Number of pairs: ~1500. 
    # For each pair, check 54 points. 1500 * 54 = 81000 ops. Very fast.
    
    point_set = set(map(tuple, points))
    
    for i in range(n_points):
        p1 = points[i]
        for j in range(i + 1, n_points):
            p2 = points[j]
            
            # Calculate line coefficients A*x + B*y + C = 0
            # p1 = (x1, y1), p2 = (x2, y2)
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            x1, y1 = p1
            x2, y2 = p2
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            # Normalize
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure first non-zero is positive
            if A < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0 and C < 0:
                A, B, C = -A, -B, -C
                
            key = (A, B, C)
            
            # Count points on this line
            # We can just increment a counter if we process all pairs? 
            # No, multiple pairs define the same line.
            # We need the *total* number of points on the line.
            # So we should calculate the count once per unique line.
            
            if key not in line_counts:
                # Count all points on this line
                count = 0
                for p in points:
                    if A * p[0] + B * p[1] + C == 0:
                        count += 1
                line_counts[key] = count
            # If key exists, we don't need to recount, the count is the same for all pairs on that line.
            # But we must ensure we don't miss a line that has only 1 point? 
            # No, we only care about lines with >= 2 points to define them, but the count will be >= 2.
            # Lines with < 2 points have count 0 or 1, which won't be in the top 27.
            
    # 3. Select Top 27 Lines
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    
    # We need 27 distinct lines.
    # The keys are distinct by construction.
    best_lines = []
    total_inc = 0
    for i in range(min(27, len(sorted_lines))):
        line_key, count = sorted_lines[i]
        best_lines.append(line_key)
        total_inc += count
        
    if len(best_lines) < 27:
        # Should not happen with 54 points
        return None, None, 0
        
    lines_array = np.array(best_lines, dtype=np.int64)
    
    return points, lines_array, total_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Strategy 1: 8x7 Grid with 2 removals
    W, H = 8, 7
    grid_points = [(x, y) for y in range(H) for x in range(W)]
    
    # Try all pairs of removals
    # 56 points -> 1540 pairs
    for p1, p2 in combinations(grid_points, 2):
        if time.time() > deadline:
            break
            
        removals = {p1, p2}
        pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
        
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lns
            
    # Strategy 2: 9x6 Grid (No removals)
    W, H = 9, 6
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 3: 6x9 Grid (No removals) - Symmetric to 9x6 but good to check
    W, H = 6, 9
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 4: 10x5 Grid (No removals)
    W, H = 10, 5
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 5: 12x4 Grid (No removals)
    W, H = 12, 4
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 6: 18x3 Grid (No removals)
    W, H = 18, 3
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 7: 27x2 Grid (No removals)
    W, H = 27, 2
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    # Strategy 8: 54x1 Grid (No removals)
    W, H = 54, 1
    removals = set()
    pts, lns, score = get_lines_and_points_for_grid(W, H, removals)
    if score > best_score:
        best_score = score
        best_points = pts
        best_lines = lns
        
    if best_points is None:
        # Fallback to standard construction
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
        
    return best_points, best_lines
