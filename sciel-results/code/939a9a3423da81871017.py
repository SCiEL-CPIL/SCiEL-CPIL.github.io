import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """
    Normalize a line (A, B, C) such that:
    1. gcd(|A|, |B|, |C|) == 1
    2. The first non-zero entry is positive.
    Returns (A, B, C) or None if invalid.
    """
    if A == 0 and B == 0:
        return None
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count incidences between points (N, 2) and lines (M, 3).
    """
    count = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A*x + B*y + C == 0:
                count += 1
    return count

def szemeredi_trotter_incidences():
    k = 3
    # Initial grid
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidences(best_points, best_lines)
    
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    # Greedy Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try moving points
        for i in range(len(points)):
            if time.time() > deadline:
                break
            x, y = points[i]
            # Try 8 neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    new_point = np.array([nx, ny])
                    
                    # Check uniqueness
                    if np.any(np.all(points == new_point, axis=1)):
                        continue
                    
                    # Calculate change in incidences
                    old_count = 0
                    new_count = 0
                    for l in lines:
                        A, B, C = l
                        if A*x + B*y + C == 0:
                            old_count += 1
                        if A*nx + B*ny + C == 0:
                            new_count += 1
                    
                    if new_count > old_count:
                        points[i] = new_point
                        improved = True
                        # Update best if better
                        current_score = _count_incidences(points, lines)
                        if current_score > best_score:
                            best_score = current_score
                            best_points = points.copy()
                            best_lines = lines.copy()
                        break # Restart point loop? Or just continue to next point?
                        # Breaking inner loops is good to restart evaluation from new state
                        # But for efficiency, we can just accept and continue.
                        # Let's accept and break out of dx/dy loops for this point.
        
        # 2. Try modifying lines
        for j in range(len(lines)):
            if time.time() > deadline:
                break
            A, B, C = lines[j]
            
            # Generate candidate lines
            candidates = [
                (A, B, C + 1),
                (A, B, C - 1),
                (A + 1, B, C),
                (A - 1, B, C),
                (A, B + 1, C),
                (A, B - 1, C),
                (A + 1, B + 1, C),
                (A - 1, B - 1, C),
            ]
            
            for cand in candidates:
                norm_cand = _normalize_line(*cand)
                if norm_cand is None:
                    continue
                
                # Check if line is already in the set
                if np.any(np.all(lines == np.array(norm_cand), axis=1)):
                    continue
                
                # Calculate change in incidences
                old_count = 0
                new_count = 0
                for p in points:
                    x, y = p
                    if A*x + B*y + C == 0:
                        old_count += 1
                    A2, B2, C2 = norm_cand
                    if A2*x + B2*y + C2 == 0:
                        new_count += 1
                
                if new_count > old_count:
                    lines[j] = np.array(norm_cand, dtype=np.int64)
                    improved = True
                    current_score = _count_incidences(points, lines)
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break # Break candidate loop for this line
        
    return best_points, best_lines
