# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import product


def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1].astype(np.int64)  # (n,1)
    py = points[:, 1:2].astype(np.int64)  # (n,1)
    A = lines[:, 0:1].astype(np.int64).T   # (1,m)
    B = lines[:, 1:2].astype(np.int64).T   # (1,m)
    C = lines[:, 2:3].astype(np.int64).T   # (1,m)
    total = px * A + py * B + C  # (n,m)
    return int(np.sum(total == 0))


def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(gcd(abs(int(A)), abs(int(B))), abs(int(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    # Make first nonzero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_key(line):
    """Get normalized key for a line tuple (A,B,C)."""
    return _normalize_line(*line)


def _compute_all_incidences(points, candidates):
    """Compute incidence counts for all candidate lines against points.
    Returns array of shape (num_candidates,) with incidence counts."""
    n = len(points)
    m = len(candidates)
    if n == 0 or m == 0:
        return np.zeros(m, dtype=np.int64)
    px = points[:, 0:1].astype(np.int64)  # (n,1)
    py = points[:, 1:2].astype(np.int64)  # (n,1)
    A = candidates[:, 0:1].astype(np.int64).T  # (1,m)
    B = candidates[:, 1:2].astype(np.int64).T  # (1,m)
    C = candidates[:, 2:3].astype(np.int64).T  # (1,m)
    total = px * A + py * B + C  # (n,m)
    return np.sum(total == 0, axis=0)


def _local_search(points, selected_lines, all_candidates, max_iters=500, time_limit=None, start_time=None):
    """Local search: try swapping one selected line for one unselected to improve total incidences."""
    if start_time is None:
        start_time = time.time()
    n_lines = len(selected_lines)
    selected_set = set(selected_lines)
    
    # Precompute incidence for each candidate
    cand_arr = np.array(all_candidates, dtype=np.int64)
    inc_counts = _compute_all_incidences(points, cand_arr)
    
    # Map from line tuple to index
    line_to_idx = {}
    for i, line in enumerate(all_candidates):
        line_to_idx[line] = i
    
    current_score = _incidence_count(points, np.array(selected_lines, dtype=np.int64))
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if time_limit is not None and time.time() > start_time + time_limit:
            break
        improved = False
        iters += 1
        
        best_gain = 0
        best_swap = None  # (selected_idx, new_line)
        
        for sel_idx, sel_line in enumerate(selected_lines):
            sel_inc = inc_counts[line_to_idx[sel_line]]
            for new_line in all_candidates:
                if new_line in selected_set:
                    continue
                new_inc = inc_counts[line_to_idx[new_line]]
                gain = new_inc - sel_inc
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (sel_idx, new_line)
        
        if best_swap is not None:
            sel_idx, new_line = best_swap
            old_line = selected_lines[sel_idx]
            selected_set.remove(old_line)
            selected_set.add(new_line)
            selected_lines[sel_idx] = new_line
            current_score += best_gain
            improved = True
    
    return selected_lines, current_score


def _local_search_points(points, selected_lines, all_candidates, max_iters=100, time_limit=None, start_time=None):
    """Local search: try moving one point to a nearby location to improve total incidences."""
    if start_time is None:
        start_time = time.time()
    n_pts = len(points)
    lines_arr = np.array(selected_lines, dtype=np.int64)
    
    # Precompute incidence: which lines each point is on
    # inc_matrix[i,j] = 1 if point i is on line j
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = lines_arr[:, 0:1].astype(np.int64).T
    B = lines_arr[:, 1:2].astype(np.int64).T
    C = lines_arr[:, 2:3].astype(np.int64).T
    inc_matrix = (px * A + py * B + C) == 0  # (n_pts, n_lines)
    
    current_score = int(np.sum(inc_matrix))
    improved = True
    iters = 0
    best_points = points.copy()
    
    while improved and iters < max_iters:
        if time_limit is not None and time.time() > start_time + time_limit:
            break
        improved = False
        iters += 1
        
        best_gain = 0
        best_move = None  # (pt_idx, new_x, new_y)
        
        for i in range(n_pts):
            x, y = points[i]
            # Try nearby positions
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 1 or ny < 1:
                        continue
                    # Check if new position is already occupied
                    # (we'll handle duplicates later)
                    # Compute new incidences for this point
                    new_vals = A[0] * nx + B[0] * ny + C[0]  # (1, n_lines)
                    new_incs = (new_vals == 0).sum()
                    old_incs = int(inc_matrix[i].sum())
                    gain = new_incs - old_incs
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, nx, ny)
        
        if best_move is not None:
            pt_idx, nx, ny = best_move
            # Check for duplicate
            pts_copy = best_points.copy()
            pts_copy[pt_idx] = [nx, ny]
            # Check no duplicates
            if len(np.unique(pts_copy, axis=0)) == n_pts:
                best_points[pt_idx] = [nx, ny]
                # Update inc_matrix
                A_col = lines_arr[:, 0]
                B_col = lines_arr[:, 1]
                C_col = lines_arr[:, 2]
                new_vals = A_col * nx + B_col * ny + C_col
                inc_matrix[pt_idx] = (new_vals == 0)
                current_score += best_gain
                improved = True
            else:
                # Try other moves
                improved = False
    
    return best_points, current_score


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Grid dimensions to try: (width, height) where width*height=54
    grid_dims = [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]
    
    for w, h in grid_dims:
        if time.time() > deadline - 60:
            break
        
        # Create grid points
        xs = np.arange(1, w + 1)
        ys = np.arange(1, h + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate candidate lines
        candidates = []
        
        # Horizontal lines
        for c in range(1, h + 1):
            candidates.append((0, 1, -c))
        
        # Vertical lines
        for c in range(1, w + 1):
            candidates.append((1, 0, -c))
        
        # Slope 1: y = x + b -> (1, -1, b)
        for b in range(1 - w, h):
            candidates.append((1, -1, b))
        
        # Slope -1: y = -x + b -> (1, 1, -b)
        for b in range(2, w + h + 1):
            candidates.append((1, 1, -b))
        
        # Slope 2: y = 2x + b -> (2, -1, b)
        for b in range(1 - 2 * w, h - 1):
            candidates.append((2, -1, b))
        
        # Slope 1/2: (1, -2, c) where c = 2y - x
        for c in range(1 - w, 2 * h):
            candidates.append((1, -2, c))
        
        # Slope -2: y = -2x + b -> (2, 1, -b)
        for b in range(3, h + 2 * w + 1):
            candidates.append((2, 1, -b))
        
        # Slope -1/2: (1, 2, -c) where c = x + 2y
        for c in range(3, w + 2 * h + 1):
            candidates.append((1, 2, -c))
        
        # Additional slopes for diversity
        # Slope 3: y = 3x + b -> (3, -1, b)
        for b in range(1 - 3 * w, h - 2):
            candidates.append((3, -1, b))
        
        # Slope 1/3: (1, -3, c) where c = 3y - x
        for c in range(1 - w, 3 * h):
            candidates.append((1, -3, c))
        
        # Slope -3: y = -3x + b -> (3, 1, -b)
        for b in range(4, h + 3 * w + 1):
            candidates.append((3, 1, -b))
        
        # Slope -1/3: (1, 3, -c) where c = x + 3y
        for c in range(4, w + 3 * h + 1):
            candidates.append((1, 3, -c))
        
        # Remove duplicates and invalid lines
        seen = set()
        unique_candidates = []
        for line in candidates:
            key = _line_key(line)
            if key == (0, 0, 0):
                continue
            if key not in seen:
                seen.add(key)
                unique_candidates.append(key)
        
        # Greedily select 27 lines using vectorized incidence
        cand_arr = np.array(unique_candidates, dtype=np.int64)
        inc_counts = _compute_all_incidences(points, cand_arr)
        
        selected = []
        selected_set = set()
        
        for _ in range(27):
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_inc = -1
            for i, line in enumerate(unique_candidates):
                if line in selected_set:
                    continue
                if inc_counts[i] > best_inc:
                    best_inc = inc_counts[i]
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(unique_candidates[best_idx])
            selected_set.add(unique_candidates[best_idx])
        
        if len(selected) == 27:
            # Local search on line selection
            remaining_time = max(10, deadline - time.time() - 30)
            selected, score = _local_search(
                points, selected, unique_candidates,
                max_iters=200, time_limit=remaining_time, start_time=start_time
            )
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = np.array(selected, dtype=np.int64)
    
    # If we didn't find a good solution, fall back to seed
    if best_points is None:
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


# EVOLVE-BLOCK-END
