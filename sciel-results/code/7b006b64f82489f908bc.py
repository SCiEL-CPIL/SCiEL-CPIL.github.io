import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so gcd(|A|,|B|,|C|) = 1 and first nonzero is positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Count (p,l) pairs where A*x + B*y + C == 0."""
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T  # (1, n)
    y = points[:, 1:2].T  # (1, n)
    val = A * x + B * y + C  # (m, n)
    return int(np.count_nonzero(val == 0))

def _build_initial_grid():
    """
    Build 54 points as a 9x6 grid and 27 lines.
    """
    cols = 9
    rows = 6
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    
    # Generate all lines passing through at least 2 points
    n_pts = len(points)
    line_set = set()
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            line_set.add(norm)
            
    # Now count incidences for each unique line
    lines_list = list(line_set)
    counts = []
    for A, B, C in lines_list:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.count_nonzero(val == 0))
        counts.append(cnt)
        
    counts = np.array(counts)
    # Select top 27 lines
    top_idx = np.argsort(-counts)[:27]
    selected_lines = [lines_list[i] for i in top_idx]
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

def _is_valid(points, lines):
    """Quick validity check."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    # Check for duplicate points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
        
    # Check for duplicate lines
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        A, B, C = int(A), int(B), int(C)
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _local_search(points, lines, deadline):
    """
    Local search with incremental delta scoring.
    """
    points = points.copy()
    lines = lines.copy()
    
    # Precompute incidence matrix M (27, 54) boolean
    def compute_matrix(points, lines):
        A = lines[:, 0:1]
        B = lines[:, 1:2]
        C = lines[:, 2:3]
        x = points[:, 0:1].T
        y = points[:, 1:2].T
        val = A * x + B * y + C
        return (val == 0)
        
    M = compute_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Randomization for tie-breaking and exploration
    rng = np.random.default_rng(42)
    
    # Precompute normalized lines for fast duplicate checking
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # 1. Try moving a random point
        if rng.random() < 0.7:
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            
            # Try a few random small moves
            for _ in range(5):
                dx = rng.integers(-3, 4)
                dy = rng.integers(-3, 4)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = orig_x + dx, orig_y + dy
                
                # Check duplicate points
                is_dup = False
                for i in range(54):
                    if points[i][0] == new_x and points[i][1] == new_y and i != p_idx:
                        is_dup = True
                        break
                if is_dup:
                    continue
                    
                # Calculate delta
                old_inc = int(np.sum(M[:, p_idx]))
                
                # New incidences: check new point against all lines
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > 0:
                    # Accept move
                    points[p_idx] = [new_x, new_y]
                    # Update M column
                    for l_idx in range(27):
                        A, B, C = lines[l_idx]
                        M[l_idx, p_idx] = (A * new_x + B * new_y + C == 0)
                    
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break # Try next point/line iteration
                    
        # 2. Try replacing a random line
        else:
            l_idx = rng.integers(0, 27)
            
            # Generate a new candidate line
            # Pick 2 random distinct points
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
                
            x1, y1 = points[p1_idx]
            x2, y2 = points[p2_idx]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            
            # Check if this line is already in the set
            is_dup_line = False
            for i in range(27):
                if i == l_idx:
                    continue
                if line_norms[i] == norm:
                    is_dup_line = True
                    break
            
            if is_dup_line:
                continue
                
            # Calculate delta
            old_inc = int(np.sum(M[l_idx, :]))
            
            # New incidences: check new line against all points
            new_inc = 0
            for p_idx in range(54):
                x, y = points[p_idx]
                if A * x + B * y + C == 0:
                    new_inc += 1
                    
            delta = new_inc - old_inc
            if delta > 0:
                # Accept move
                lines[l_idx] = norm
                line_norms[l_idx] = norm
                # Update M row
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (A * x + B * y + C == 0)
                    
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines, best_score

def _restart_search(points, lines, deadline, seed_offset=0):
    """
    Run local search with a different random seed for diversity.
    """
    points = points.copy()
    lines = lines.copy()
    
    def compute_matrix(points, lines):
        A = lines[:, 0:1]
        B = lines[:, 1:2]
        C = lines[:, 2:3]
        x = points[:, 0:1].T
        y = points[:, 1:2].T
        val = A * x + B * y + C
        return (val == 0)
        
    M = compute_matrix(points, lines)
    current_score = int(np.sum(M))
    best_score = current_score
    best_points = points.copy()
    best_lines = lines.copy()
    
    rng = np.random.default_rng(42 + seed_offset)
    
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        # 1. Try moving a random point
        if rng.random() < 0.7:
            p_idx = rng.integers(0, 54)
            orig_x, orig_y = points[p_idx]
            
            for _ in range(5):
                dx = rng.integers(-4, 5)
                dy = rng.integers(-4, 5)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = orig_x + dx, orig_y + dy
                
                is_dup = False
                for i in range(54):
                    if points[i][0] == new_x and points[i][1] == new_y and i != p_idx:
                        is_dup = True
                        break
                if is_dup:
                    continue
                    
                old_inc = int(np.sum(M[:, p_idx]))
                new_inc = 0
                for l_idx in range(27):
                    A, B, C = lines[l_idx]
                    if A * new_x + B * new_y + C == 0:
                        new_inc += 1
                        
                delta = new_inc - old_inc
                if delta > 0:
                    points[p_idx] = [new_x, new_y]
                    for l_idx in range(27):
                        A, B, C = lines[l_idx]
                        M[l_idx, p_idx] = (A * new_x + B * new_y + C == 0)
                    
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break
                    
        # 2. Try replacing a random line
        else:
            l_idx = rng.integers(0, 27)
            
            p1_idx = rng.integers(0, 54)
            p2_idx = rng.integers(0, 54)
            if p1_idx == p2_idx:
                continue
                
            x1, y1 = points[p1_idx]
            x2, y2 = points[p2_idx]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
                
            norm = _normalize_line(A, B, C)
            
            is_dup_line = False
            for i in range(27):
                if i == l_idx:
                    continue
                if line_norms[i] == norm:
                    is_dup_line = True
                    break
            
            if is_dup_line:
                continue
                
            old_inc = int(np.sum(M[l_idx, :]))
            new_inc = 0
            for p_idx in range(54):
                x, y = points[p_idx]
                if A * x + B * y + C == 0:
                    new_inc += 1
                    
            delta = new_inc - old_inc
            if delta > 0:
                lines[l_idx] = norm
                line_norms[l_idx] = norm
                for p_idx in range(54):
                    x, y = points[p_idx]
                    M[l_idx, p_idx] = (A * x + B * y + C == 0)
                    
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    best_lines = lines.copy()

    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580  # Safety margin
    
    # 1. Initialize with Grid
    points, lines = _build_initial_grid()
    
    init_score = _incidence_count(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = init_score
    
    # 2. Local Search with multiple restarts
    # First run with seed 0
    if time.time() < deadline - 30:
        pts, lns, score = _local_search(points, lines, deadline - 30)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # Second run with different seed and starting from best
    if time.time() < deadline - 10:
        pts, lns, score = _restart_search(best_points, best_lines, deadline - 10, seed_offset=100)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # Final check
    if not _is_valid(best_points, best_lines):
        # Return the initial valid one if search broke it
        return points, lines
        
    return best_points, best_lines
