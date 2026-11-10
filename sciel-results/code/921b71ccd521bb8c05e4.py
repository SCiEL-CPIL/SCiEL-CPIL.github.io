import numpy as np
from math import gcd
import time

def _normalize_line(A, B, C):
    """Normalize line so first nonzero of (A,B,C) is positive, divided by gcd."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    """Get line through two points as normalized (A,B,C)."""
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _count_incident_points(lines, points):
    """For each line, count how many points lie on it. Returns array of counts."""
    # lines: (m, 3) array, points: (n, 2) array
    # Compute A*x + B*y + C for all pairs
    # lines[:, 0:1] * points[:, 0:1] + lines[:, 1:2] * points[:, 1:2] + lines[:, 2:3]
    A = lines[:, 0:1].astype(np.int64)
    B = lines[:, 1:2].astype(np.int64)
    C = lines[:, 2:3].astype(np.int64)
    x = points[:, 0:1].astype(np.int64)
    y = points[:, 1:2].astype(np.int64)
    vals = A * x.T + B * y.T + C  # shape (m, n)
    return (vals == 0).sum(axis=1)

def _evaluate_grid(nx, ny):
    """Evaluate a rectangular grid of size nx x ny, return (incidence_count, points, lines)."""
    if nx * ny != 54:
        return 0, None, None
    
    points = np.array([(x, y) for x in range(nx) for y in range(ny)], dtype=np.int64)
    n = len(points)
    
    # Generate all unique lines through pairs of points
    line_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(tuple(points[i]), tuple(points[j]))
            line_set.add(line)
    
    unique_lines = list(line_set)
    if not unique_lines:
        return 0, points, None
    
    lines_arr = np.array(unique_lines, dtype=np.int64)
    
    # Count incidences for each line
    counts = _count_incident_points(lines_arr, points)
    
    # Sort by count descending, take top 27
    order = np.argsort(-counts)
    top_indices = order[:27]
    top_lines = lines_arr[top_indices]
    
    total_inc = int(counts[top_indices].sum())
    
    return total_inc, points, top_lines

def _local_search_lines(points, lines, counts, time_limit):
    """Try to improve by swapping lines. Returns (new_count, new_lines)."""
    n_lines = len(lines)
    n_all = len(counts)
    best_count = int(counts.sum())
    best_lines = lines.copy()
    start = time.time()
    
    # Build set of selected line indices
    selected = set(range(n_lines))
    
    # Try single swaps: remove one selected, add one unselected
    unselected = set(range(n_all)) - selected
    
    # Precompute: for each unselected line, how many points it hits
    # counts already has this info
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_swap_gain = 0
        best_swap = None
        
        for rem in selected:
            for add in unselected:
                gain = counts[add] - counts[rem]
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (rem, add)
        
        if best_swap and best_swap_gain > 0:
            rem, add = best_swap
            selected.discard(rem)
            selected.add(add)
            unselected.discard(add)
            unselected.add(rem)
            best_count += best_swap_gain
            best_lines = np.array([lines[i] for i in sorted(selected)], dtype=np.int64)
            improved = True
    
    return best_count, best_lines

def _local_search_points(points, lines, time_limit):
    """Try to improve by moving points. Returns (new_count, new_points, new_lines)."""
    n = len(points)
    best_count = 0
    best_points = points.copy()
    best_lines = lines.copy()
    start = time.time()
    
    # Compute current incidence
    lines_arr = lines
    counts = _count_incident_points(lines_arr, best_points)
    best_count = int(counts.sum())
    
    # Try moving each point to nearby integer positions
    for i in range(n):
        if time.time() > time_limit:
            break
        orig_x, orig_y = best_points[i]
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = orig_x + dx, orig_y + dy
                # Check if this position is already occupied
                if any(best_points[j][0] == new_x and best_points[j][1] == new_y for j in range(n) if j != i):
                    continue
                # Try this move
                old_point = best_points[i].copy()
                best_points[i] = [new_x, new_y]
                
                # Recompute incidences
                counts_new = _count_incident_points(lines_arr, best_points)
                new_count = int(counts_new.sum())
                
                if new_count > best_count:
                    best_count = new_count
                else:
                    best_points[i] = old_point
    
    return best_count, best_points, best_lines

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization with 54 points and 27 lines.
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin (evaluator kills at 630s)
    
    best_count = 0
    best_points = None
    best_lines = None
    
    # Factor pairs of 54 - try all rectangular grids
    grid_shapes = [
        (1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)
    ]
    
    for (nx, ny) in grid_shapes:
        if time.time() > deadline:
            break
        count, points, lines = _evaluate_grid(nx, ny)
        if count > best_count:
            best_count = count
            best_points = points
            best_lines = lines
    
    # If we have time, try local search to improve
    if best_points is not None and best_lines is not None and time.time() < deadline - 60:
        # Local search on lines
        lines_arr = best_lines
        counts = _count_incident_points(lines_arr, best_points)
        # We need all lines to do line swap, so regenerate
        n = len(best_points)
        line_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                line = _line_from_points(tuple(best_points[i]), tuple(best_points[j]))
                line_set.add(line)
        all_lines = np.array(list(line_set), dtype=np.int64)
        all_counts = _count_incident_points(all_lines, best_points)
        
        if time.time() < deadline - 30:
            new_count, new_lines = _local_search_lines(best_points, best_lines, all_counts, deadline - 10)
            if new_count > best_count:
                best_count = new_count
                best_lines = new_lines
    
    # Fallback
    if best_points is None:
        best_count, best_points, best_lines = _evaluate_grid(6, 9)
    
    return np.array(best_points, dtype=np.int64), np.array(best_lines, dtype=np.int64)
