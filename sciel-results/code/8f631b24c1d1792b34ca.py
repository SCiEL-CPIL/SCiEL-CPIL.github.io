# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy:
    1. Define a candidate set of grid shapes (a, b) such that a*b <= 54.
    2. For each shape, construct the grid points.
    3. If a*b < 54, add extra points (e.g., in a pattern that doesn't break symmetry too much, 
       or just try full grids where a*b=54).
       Actually, the task requires exactly 54 points.
       We will focus on full grids where a*b=54.
       Shapes: (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1).
    4. For the best shape, enumerate all lines through pairs of points.
    5. Select the 27 lines with the highest incidence counts.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    # Candidate grid dimensions (a, b) where a*b = 54
    candidates = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for a, b in candidates:
        # Generate points
        xs = np.arange(1, a + 1)
        ys = np.arange(1, b + 1)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        if points.shape != (54, 2):
            continue
            
        # Generate all candidate lines
        # We need to find the 27 best lines.
        # First, identify grid lines (vertical and horizontal) as they are strong candidates.
        # But it's safer to just enumerate all lines through pairs.
        
        n_pts = 54
        point_set = set(map(tuple, points))
        
        candidate_lines = {}
        
        # Optimization: Only consider lines that pass through at least 2 points.
        # Iterate through pairs
        for i in range(n_pts):
            p1 = points[i]
            for j in range(i + 1, n_pts):
                p2 = points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                A = dy
                B = -dx
                C = dx * p1[1] - dy * p1[0]
                
                # Normalize
                g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A, B, C) != (0, 0, 0) else 1
                if g > 0:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key not in candidate_lines:
                    candidate_lines[key] = 0
        
        # Count incidences
        for key in candidate_lines:
            A, B, C = key
            count = 0
            for pt in points:
                if A * pt[0] + B * pt[1] + C == 0:
                    count += 1
            candidate_lines[key] = count
            
        # Sort lines by count
        sorted_lines = sorted(candidate_lines.items(), key=lambda x: -x[1])
        
        # Pick top 27
        selected_lines = [l for l, c in sorted_lines[:27]]
        
        lines = np.array(selected_lines, dtype=np.int64)
        
        # Calculate total incidences
        total_inc = 0
        for pt in points:
            for l in lines:
                if l[0] * pt[0] + l[1] * pt[1] + l[2] == 0:
                    total_inc += 1
                    
        if total_inc > best_score:
            best_score = total_inc
            best_points = points
            best_lines = lines
            
        # Time check
        if time.time() > deadline - 10:
            break
            
    if best_points is None:
        # Fallback
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = np.array([(1, 0, -x) for x in range(1, 7)] + [(0, 1, -y) for y in range(1, 10)] + [(1, -1, 0)]*12, dtype=np.int64)
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
