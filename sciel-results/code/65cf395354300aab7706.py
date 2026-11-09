import numpy as np
from math import gcd
import time

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Try multiple grid configurations and pick the best
    best_total = -1
    best_points = None
    best_lines = None
    
    # Configuration 1: 6x9 grid (exactly 54 points)
    configs = [
        (6, 9),
        (9, 6),
        (7, 8),  # 56 points, remove 2
        (8, 7),  # 56 points, remove 2
        (5, 11), # 55 points, remove 1
        (11, 5), # 55 points, remove 1
    ]
    
    for (rows, cols) in configs:
        if time.time() > deadline:
            break
            
        total_pts = rows * cols
        if total_pts < 54:
            continue
        if total_pts > 54 + 10:  # don't remove too many
            continue
        
        # Generate grid points
        xs = np.arange(1, cols + 1)
        ys = np.arange(1, rows + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        if total_pts == 54:
            points = all_points
        else:
            # Remove (total_pts - 54) corner points
            n_remove = total_pts - 54
            # Remove corners: (1,1), (cols,rows), (1,rows), (cols,1)
            corner_indices = []
            if n_remove >= 1:
                corner_indices.append(0)  # (1,1)
            if n_remove >= 2:
                corner_indices.append(total_pts - 1)  # (cols, rows)
            if n_remove >= 3:
                corner_indices.append(rows - 1)  # (1, rows)
            if n_remove >= 4:
                corner_indices.append(cols - 1)  # (cols, 1)
            points = np.delete(all_points, corner_indices[:n_remove], axis=0)
        
        if len(points) != 54:
            continue
        
        # Generate all lines from pairs
        n = len(points)
        lines_dict = {}  # normalized line -> incidence count
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                A = y2 - y1
                B = -(x2 - x1)
                C = (x2 - x1) * y1 - (y2 - y1) * x2
                key = _normalize_line(A, B, C)
                if key not in lines_dict:
                    lines_dict[key] = 0
        
        # Count incidences for each unique line
        for (A, B, C) in lines_dict:
            count = 0
            for x, y in points:
                if A * x + B * y + C == 0:
                    count += 1
            lines_dict[(A, B, C)] = count
        
        # Sort by incidence count descending
        sorted_lines = sorted(lines_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 27
        selected = sorted_lines[:27]
        total_inc = sum(count for _, count in selected)
        
        if total_inc > best_total:
            best_total = total_inc
            best_points = points.copy()
            best_lines = np.array([l for l, _ in selected], dtype=np.int64)
    
    if best_points is None:
        # Fallback: 6x9 grid
        xs = np.arange(1, 10)
        ys = np.arange(1, 7)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Just use all horizontal and vertical lines
        lines_list = []
        for y in range(1, 7):
            lines_list.append((0, 1, -y))
        for x in range(1, 10):
            lines_list.append((1, 0, -x))
        # Add some diagonals
        for d in range(-8, 9):
            lines_list.append((1, -1, d))
            if len(lines_list) >= 27:
                break
        best_lines = np.array(lines_list[:27], dtype=np.int64)
    
    return best_points, best_lines
