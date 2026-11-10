# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    """Return a canonical integer triple for the line A*x + B*y + C = 0."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    # Make the first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _build_grid(m, n):
    xs = np.arange(1, m + 1, dtype=np.int64)
    ys = np.arange(1, n + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _find_best_lines_for_points(points, K, deadline):
    """
    For a fixed point set, find the 27 lines with highest incidence
    by scanning line space (A,B) with gcd(A,B)=1, then finding best C.
    """
    n_pts = len(points)
    px = points[:, 0]
    py = points[:, 1]
    
    candidates = []  # List of (count, normalized_line_tuple)
    
    for A in range(-K, K + 1):
        if time.time() > deadline:
            break
        for B in range(-K, K + 1):
            if A == 0 and B == 0:
                continue
            if gcd(abs(A), abs(B)) != 1:
                continue
            
            # Compute values V = A*x + B*y
            V = A * px + B * py
            
            # For each unique value v, line is A*x + B*y + C = 0 with C = -v
            # Count = number of points with V == v
            unique_vals, counts = np.unique(V, return_counts=True)
            
            for val, count in zip(unique_vals, counts):
                if count == 0:
                    continue
                C = -val
                line_norm = _normalize(A, B, C)
                candidates.append((count, line_norm))
    
    if not candidates:
        return None, 0
    
    # Sort by count descending, then by line for determinism
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    # Select top 27 unique lines
    selected_lines = []
    seen = set()
    total_incidence = 0
    
    for count, line_norm in candidates:
        if len(selected_lines) >= 27:
            break
        if line_norm not in seen:
            seen.add(line_norm)
            selected_lines.append(line_norm)
            total_incidence += count
    
    if len(selected_lines) < 27:
        # Fallback: fill with any valid lines
        for A in range(-20, 21):
            if len(selected_lines) >= 27:
                break
            for B in range(-20, 21):
                if len(selected_lines) >= 27:
                    break
                if A == 0 and B == 0:
                    continue
                if gcd(abs(A), abs(B)) != 1:
                    continue
                for C in range(-100, 101):
                    norm = _normalize(A, B, C)
                    if norm not in seen:
                        seen.add(norm)
                        selected_lines.append(norm)
                        # Compute actual incidence
                        val = A * px + B * py + C
                        total_incidence += int(np.sum(val == 0))
                        if len(selected_lines) >= 27:
                            break
    
    if len(selected_lines) < 27:
        return None, 0
    
    lines_array = np.array(selected_lines, dtype=np.int64)
    
    # Recompute total incidence to be safe
    total_incidence = 0
    for line in selected_lines:
        A, B, C = line
        val = A * px + B * py + C
        total_incidence += int(np.sum(val == 0))
    
    return lines_array, total_incidence


def _refine_points(points, lines, max_iterations=50, deadline=None):
    """
    Local search: try moving each point to a nearby integer position
    to increase total incidence.
    """
    px = points[:, 0].copy()
    py = points[:, 1].copy()
    n_pts = len(points)
    
    best_incidence = 0
    for line in lines:
        A, B, C = line
        val = A * px + B * py + C
        best_incidence += int(np.sum(val == 0))
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        if deadline and time.time() > deadline:
            break
        
        for i in range(n_pts):
            if deadline and time.time() > deadline:
                break
            
            best_local_inc = best_incidence
            best_pos = (px[i], py[i])
            
            # Try nearby positions (radius 3)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = px[i] + dx
                    ny = py[i] + dy
                    
                    # Check if this position is already occupied
                    if np.any((px == nx) & (py == ny)):
                        continue
                    
                    # Compute new incidence
                    new_inc = 0
                    for line in lines:
                        A, B, C = line
                        val = A * px + B * py + C
                        # Update point i
                        old_val = val[i]
                        new_val = A * nx + B * ny + C
                        val[i] = new_val
                        new_inc += int(np.sum(val == 0))
                        val[i] = old_val
                    
                    if new_inc > best_local_inc:
                        best_local_inc = new_inc
                        best_pos = (nx, ny)
            
            if best_pos != (px[i], py[i]):
                px[i], py[i] = best_pos
                best_incidence = best_local_inc
                improved = True
    
    new_points = np.stack([px, py], axis=1).astype(np.int64)
    return new_points, best_incidence


def _refine_lines(points, lines, K, max_iterations=30, deadline=None):
    """
    Local search: try replacing each line with a better one from the line space.
    """
    px = points[:, 0]
    py = points[:, 1]
    n_pts = len(points)
    
    # Compute current incidence per line
    line_incs = []
    for line in lines:
        A, B, C = line
        val = A * px + B * py + C
        line_incs.append(int(np.sum(val == 0)))
    
    total_inc = sum(line_incs)
    
    improved = True
    iters = 0
    while improved and iters < max_iterations:
        improved = False
        iters += 1
        if deadline and time.time() > deadline:
            break
        
        for idx in range(27):
            if deadline and time.time() > deadline:
                break
            
            # Find the best replacement line (not already in the set)
            current_line = tuple(lines[idx])
            current_inc = line_incs[idx]
            
            # Scan a subset of line space
            best_new_line = None
            best_new_inc = current_inc
            
            for A in range(-K, K + 1):
                if deadline and time.time() > deadline:
                    break
                for B in range(-K, K + 1):
                    if A == 0 and B == 0:
                        continue
                    if gcd(abs(A), abs(B)) != 1:
                        continue
                    
                    V = A * px + B * py
                    unique_vals, counts = np.unique(V, return_counts=True)
                    
                    for val, count in zip(unique_vals, counts):
                        if count <= best_new_inc:
                            continue
                        C = -val
                        line_norm = _normalize(A, B, C)
                        
                        # Check if this line is already in the set
                        if line_norm in [tuple(lines[j]) for j in range(27) if j != idx]:
                            continue
                        
                        best_new_line = line_norm
                        best_new_inc = count
            
            if best_new_line is not None and best_new_inc > current_inc:
                lines[idx] = best_new_line
                line_incs[idx] = best_new_inc
                total_inc += best_new_inc - current_inc
                improved = True
    
    return lines, total_inc


def szemeredi_trotter_incidences():
    """
    Combined approach:
    1. Try multiple grid configurations
    2. For each, use line-space search to find best 27 lines
    3. Refine points via local search
    4. Refine lines via local search
    5. Return the best (points, lines) found
    """
    t_start = time.time()
    deadline = t_start + 570.0  # Safety margin before 630s kill
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate grid dimensions
    grid_configs = [
        (6, 9), (9, 6),
        (7, 8), (8, 7),
        (5, 11), (11, 5),
        (4, 14), (14, 4),
        (3, 18), (18, 3),
        (2, 27), (27, 2),
    ]
    
    K = 30  # Max slope component for line search
    
    for w, h in grid_configs:
        if time.time() > deadline - 60:
            break
        
        points = _build_grid(w, h)
        
        # Ensure exactly 54 points
        if len(points) > 54:
            points = points[:54]
        elif len(points) < 54:
            continue
        
        # Find best lines for this point set
        lines, score = _find_best_lines_for_points(points, K, deadline - 30)
        
        if lines is None:
            continue
        
        # Refine points
        if time.time() < deadline - 45:
            points_refined, score_refined = _refine_points(
                points, lines, max_iterations=30, deadline=deadline - 30
            )
            if score_refined > score:
                points = points_refined
                score = score_refined
        
        # Refine lines
        if time.time() < deadline - 30:
            lines_refined, score_refined2 = _refine_lines(
                points, lines, K // 2, max_iterations=20, deadline=deadline - 15
            )
            if score_refined2 > score:
                lines = lines_refined
                score = score_refined2
        
        if score > best_score:
            best_score = score
            best_points = points
            best_lines = lines
    
    # If no grid worked, fallback
    if best_points is None:
        points = _build_grid(6, 9)
        lines = np.array([
            [1, 0, -1], [1, 0, -2], [1, 0, -3], [1, 0, -4], [1, 0, -5], [1, 0, -6],
            [0, 1, -1], [0, 1, -2], [0, 1, -3], [0, 1, -4], [0, 1, -5], [0, 1, -6],
            [1, 1, -2], [1, 1, -3], [1, 1, -4], [1, 1, -5], [1, 1, -6], [1, 1, -7],
            [1, -1, 0], [1, -1, 1], [1, -1, 2], [1, -1, 3], [1, -1, 4], [1, -1, 5],
            [-1, 1, 0], [-1, 1, -1], [-1, 1, -2],
        ], dtype=np.int64)
        best_points = points
        best_lines = lines
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
