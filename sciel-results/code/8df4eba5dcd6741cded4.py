import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _count_incidentences(points, lines):
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _score_point_set(points):
    # 1. Generate all lines passing through at least 2 points
    # To keep it fast, we limit the search space or use a heuristic if N is large, 
    # but 54 points -> 1431 pairs is manageable.
    line_counts = {}
    n = len(points)
    
    # Optimization: Only consider lines with count >= 2? 
    # Actually, a line with count 1 contributes 1. 
    # If we have 27 lines, and we pick top 27, we might pick lines with count 1 if there aren't enough lines with count >= 2.
    # But usually, in these problems, lines with count >= 2 are much better.
    # Let's generate all lines from pairs.
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            if key not in line_counts:
                line_counts[key] = 0
            # Increment count for points on this line
            # This is O(N) per line, so O(N^3) total. 54^3 = 157k ops. Very fast.
            
    # Actually, counting incidence for each unique line is O(N * M_unique).
    # M_unique can be up to 1431. 54 * 1431 = 77k ops.
    
    counts = []
    for key in line_counts.keys():
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        counts.append(cnt)
        
    # Sort counts descending
    counts.sort(reverse=True)
    
    # Sum top 27
    total = sum(counts[:27])
    return total

def szemeredi_trotter_incidences():
    deadline = time.time() + 580 # Safety margin
    start_time = time.time()
    
    # Initial configuration: 6x9 grid
    xs = list(range(1, 7))
    ys = list(range(1, 10))
    points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
    
    best_points = points.copy()
    best_score = _score_point_set(points)
    current_points = points.copy()
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.99
    T_min = 0.1
    
    # We will perform moves:
    # 1. Move a random point to a random nearby location (delta x, delta y in -2..2)
    # 2. Swap two points? No, points are distinct.
    # 3. Move a point to a random location within a bounding box?
    
    # To ensure validity (distinct integer points), we maintain a set of occupied points.
    occupied = set((int(p[0]), int(p[1])) for p in current_points)
    
    # We need to be careful with the "nearby" logic to keep points within a reasonable range
    # so that line coefficients don't get too huge (though int64 handles it).
    # Let's say we restrict points to [-20, 20] x [-20, 20] for the search space.
    
    while time.time() < deadline and T > T_min:
        # Pick a random point to move
        idx = np.random.randint(0, 54)
        old_x, old_y = current_points[idx]
        old_pos = (int(old_x), int(old_y))
        
        # Generate new position
        # Try a few random offsets
        moved = False
        for _ in range(10):
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            new_x = int(old_x) + dx
            new_y = int(old_y) + dy
            new_pos = (new_x, new_y)
            
            if new_pos == old_pos:
                continue
            if new_pos in occupied:
                continue
            # Keep within bounds
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                # Perform move
                occupied.remove(old_pos)
                occupied.add(new_pos)
                current_points[idx] = [new_x, new_y]
                moved = True
                break
        
        if not moved:
            T *= cooling_rate
            continue
            
        # Evaluate new score
        new_score = _score_point_set(current_points)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Revert
            occupied.remove(new_pos)
            occupied.add(old_pos)
            current_points[idx] = [int(old_x), int(old_y)]
            
        T *= cooling_rate
        
        # Periodic restart or cooling adjustment?
        # If stuck for a long time, maybe increase T?
        # For now, simple cooling.
        
    # Final check
    if not _is_valid(best_points, np.zeros((27,3), dtype=np.int64)): # Just checking points validity via a dummy line set? 
         # _is_valid checks both. Let's construct a valid line set for the best points to return.
         pass
         
    # Construct the actual lines for the best points to return
    # We need to return (points, lines)
    # We must generate the top 27 lines for best_points
    
    line_counts = {}
    n = len(best_points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = best_points[i]
            x2, y2 = best_points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            if key not in line_counts:
                line_counts[key] = 0
                
    counts = []
    for key in line_counts.keys():
        A, B, C = key
        cnt = 0
        for p in best_points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        counts.append((cnt, key))
        
    counts.sort(reverse=True, key=lambda x: x[0])
    
    top_keys = [k for (_, k) in counts[:27]]
    
    # If we have fewer than 27 lines (unlikely), pad with dummy lines?
    # The problem says "27 lines". If we don't have 27 unique lines from pairs, 
    # we might need to add lines with count 1 or 0?
    # But we want to maximize incidences.
    # If we have < 27 lines from pairs, we can add arbitrary lines that don't overlap.
    # However, 54 points define C(54,2)=1431 pairs, so we will always have > 27 lines.
    
    lines_arr = np.array(top_keys, dtype=np.int64)
    
    # Ensure validity
    if not _is_valid(best_points, lines_arr):
        # Fallback to initial grid
        points = np.array([(x, y) for x in xs for y in ys], dtype=np.int64)
        line_counts = {}
        for i in range(54):
            for j in range(i+1, 54):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                if A == 0 and B == 0: continue
                key = _line_key(A, B, C)
                if key[0] == 0 and key[1] == 0: continue
                if key not in line_counts:
                    line_counts[key] = 0
        counts = []
        for key in line_counts.keys():
            A, B, C = key
            cnt = 0
            for p in points:
                x, y = p
                if A * x + B * y + C == 0:
                    cnt += 1
            counts.append((cnt, key))
        counts.sort(reverse=True, key=lambda x: x[0])
        top_keys = [k for (_, k) in counts[:27]]
        lines_arr = np.array(top_keys, dtype=np.int64)
        return points, lines_arr
        
    return best_points, lines_arr
