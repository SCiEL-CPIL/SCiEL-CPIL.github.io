# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[:, 0:1].T
    B = lines[:, 1:2].T
    C = lines[:, 2:3].T
    total = px * A + py * B + C
    return int(np.sum(total == 0))

def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _compute_all_incidences(points, cand_array):
    """Compute incidence counts for all candidate lines given points."""
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = cand_array[:, 0:1].T
    B = cand_array[:, 1:2].T
    C = cand_array[:, 2:3].T
    total = px * A + py * B + C
    return np.sum(total == 0, axis=0)

def _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=6):
    """Generate candidate lines that intersect the bounding box."""
    candidates = []
    for A in range(-max_coef, max_coef + 1):
        for B in range(-max_coef, max_coef + 1):
            if A == 0 and B == 0:
                continue
            # Determine C range so line intersects the bounding box
            cx_min, cx_max = 0, 0
            if A > 0:
                cx_min, cx_max = -A * x_max, -A * x_min
            elif A < 0:
                cx_min, cx_max = -A * x_min, -A * x_max
            
            cy_min, cy_max = 0, 0
            if B > 0:
                cy_min, cy_max = -B * y_max, -B * y_min
            elif B < 0:
                cy_min, cy_max = -B * y_min, -B * y_max
            
            C_min = cx_min + cy_min
            C_max = cx_max + cy_max
            
            for C in range(C_min, C_max + 1):
                candidates.append((A, B, C))
    return np.array(candidates, dtype=np.int64)

def _select_best_lines(points, cand_array, k=27):
    """Select top k unique lines by incidence count."""
    incidences = _compute_all_incidences(points, cand_array)
    sorted_indices = np.argsort(incidences)[::-1]
    
    selected_lines = []
    seen_normalized = set()
    
    for idx in sorted_indices:
        if len(selected_lines) >= k:
            break
        A, B, C = cand_array[idx]
        norm = _normalize_line(A, B, C)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            selected_lines.append(norm)
    
    return np.array(selected_lines[:k], dtype=np.int64)

def _local_search_points(points, cand_array, k=27, max_iter=80, deadline=None):
    """
    Local search: try moving individual points to improve total incidence.
    Uses delta computation for speed.
    """
    best_points = points.copy()
    best_lines = _select_best_lines(best_points, cand_array, k)
    best_score = _incidence_count(best_points, best_lines)
    
    # Precompute incidence vector for all candidate lines
    all_incidences = _compute_all_incidences(best_points, cand_array)
    n_cand = len(cand_array)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        improved = False
        n_points = len(best_points)
        
        for i in range(n_points):
            old_x, old_y = int(best_points[i, 0]), int(best_points[i, 1])
            
            # Try small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Skip if this creates a duplicate point
                    if np.any((best_points[:, 0] == new_x) & (best_points[:, 1] == new_y)):
                        continue
                    
                    # Compute delta: how many candidate lines gain/lose this point
                    # A line (A,B,C) contains (x,y) iff A*x + B*y + C == 0
                    # Delta = count of lines where new point is on line - count where old point was on line
                    # For old point: lines where A*old_x + B*old_y + C == 0
                    # For new point: lines where A*new_x + B*new_y + C == 0
                    
                    # Efficient: compute for all candidates at once
                    px_old = old_x
                    py_old = old_y
                    px_new = new_x
                    py_new = new_y
                    
                    old_on = cand_array[:, 0] * px_old + cand_array[:, 1] * py_old + cand_array[:, 2]
                    new_on = cand_array[:, 0] * px_new + cand_array[:, 1] * py_new + cand_array[:, 2]
                    
                    # How many lines contain old point
                    old_count = int(np.sum(old_on == 0))
                    # How many lines contain new point
                    new_count = int(np.sum(new_on == 0))
                    
                    # But we need to check if the top-k lines change
                    # For a quick estimate, use the delta in total incidences
                    # The actual score change depends on which lines are in the top-k
                    # For now, use a heuristic: if new_count > old_count, likely improvement
                    
                    if new_count <= old_count:
                        continue
                    
                    # Temporarily move point and recompute
                    best_points[i, 0], best_points[i, 1] = new_x, new_y
                    new_lines = _select_best_lines(best_points, cand_array, k)
                    new_score = _incidence_count(best_points, new_lines)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_lines = new_lines
                        all_incidences = _compute_all_incidences(best_points, cand_array)
                        improved = True
                    else:
                        # Revert
                        best_points[i, 0], best_points[i, 1] = old_x, old_y
        
        if not improved:
            break
    
    return best_points, best_lines, best_score

def _refine_lines(points, cand_array, k=27, deadline=None):
    """
    Refine the line selection by trying to swap in better lines.
    """
    best_lines = _select_best_lines(points, cand_array, k)
    best_score = _incidence_count(points, best_lines)
    
    for iteration in range(20):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        improved = False
        incidences = _compute_all_incidences(points, cand_array)
        sorted_indices = np.argsort(incidences)[::-1]
        
        # Current normalized lines
        current_norms = set()
        for j in range(len(best_lines)):
            current_norms.add(_normalize_line(*best_lines[j]))
        
        # Try replacing each line with a better one
        for i in range(len(best_lines)):
            old_line = best_lines[i]
            old_norm = _normalize_line(*old_line)
            
            # Find incidence of old line
            old_idx = np.where((cand_array[:, 0] == old_line[0]) & 
                               (cand_array[:, 1] == old_line[1]) & 
                               (cand_array[:, 2] == old_line[2]))[0]
            old_inc = int(incidences[old_idx[0]]) if len(old_idx) > 0 else 0
            
            # Find best replacement not already in set
            for idx in sorted_indices:
                A, B, C = cand_array[idx]
                norm = _normalize_line(A, B, C)
                if norm in current_norms:
                    continue
                if incidences[idx] > old_inc:
                    # Swap
                    best_lines[i] = np.array([A, B, C], dtype=np.int64)
                    current_norms.discard(old_norm)
                    current_norms.add(norm)
                    improved = True
                    break
        
        if not improved:
            break
    
    best_score = _incidence_count(points, best_lines)
    return best_lines, best_score

def _try_grid_configurations(deadline):
    """
    Try multiple grid configurations and return the best (points, lines, score).
    """
    best_points = None
    best_lines = None
    best_score = -1
    
    # Try different grid dimensions
    configs = [
        # (nrows, ncols, remove_corners)
        (7, 8, [(0,0), (6,7)]),
        (7, 8, [(0,7), (6,0)]),
        (6, 9, [(0,0), (5,8)]),
        (6, 9, [(0,8), (5,0)]),
        (5, 11, [(0,0), (4,10)]),
        (5, 11, [(0,10), (4,0)]),
        (9, 6, [(0,0), (8,5)]),
        (9, 6, [(0,5), (8,0)]),
        (4, 14, [(0,0), (3,13)]),
        (4, 14, [(0,13), (3,0)]),
        (3, 18, [(0,0), (2,17)]),
        (3, 18, [(0,17), (2,0)]),
        (11, 5, [(0,0), (10,4)]),
        (11, 5, [(0,4), (10,0)]),
        (14, 4, [(0,0), (13,3)]),
        (14, 4, [(0,3), (13,0)]),
        (18, 3, [(0,0), (17,2)]),
        (18, 3, [(0,2), (17,0)]),
    ]
    
    for nrows, ncols, remove_corners in configs:
        if time.time() > deadline - 30:
            break
        
        # Generate grid points
        xs = np.arange(1, ncols + 1)
        ys = np.arange(1, nrows + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        
        # Create mask
        mask = np.ones((nrows, ncols), dtype=bool)
        for (ri, ci) in remove_corners:
            mask[ri, ci] = False
        
        points = np.stack([px[mask], py[mask]], axis=1).astype(np.int64)
        
        if len(points) != 54:
            continue
        
        # Generate candidate lines
        x_min, x_max = 1, ncols
        y_min, y_max = 1, nrows
        cand_array = _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=6)
        
        # Select initial best lines
        lines = _select_best_lines(points, cand_array, 27)
        score = _incidence_count(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses multiple grid configurations with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    # 1. Try multiple grid configurations
    points, lines, score = _try_grid_configurations(deadline)
    
    if points is None:
        # Fallback: 7x8 grid with 2 corners removed
        xs = np.arange(1, 8)
        ys = np.arange(1, 9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        mask = ~(((px == 1) & (py == 1)) | ((px == 7) & (py == 8)))
        points = np.stack([px[mask], py[mask]], axis=1).astype(np.int64)
        x_min, x_max = 1, 7
        y_min, y_max = 1, 8
        cand_array = _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=6)
        lines = _select_best_lines(points, cand_array, 27)
        score = _incidence_count(points, lines)
    else:
        # Regenerate candidate lines for the best configuration
        x_min = int(np.min(points[:, 0]))
        x_max = int(np.max(points[:, 0]))
        y_min = int(np.min(points[:, 1]))
        y_max = int(np.max(points[:, 1]))
        cand_array = _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=6)
    
    # 2. Local search to improve points
    if time.time() < deadline - 120:
        points, lines, score = _local_search_points(points, cand_array, 27, max_iter=60, deadline=deadline)
    
    # 3. Refine lines
    if time.time() < deadline - 30:
        lines, score = _refine_lines(points, cand_array, 27, deadline=deadline)
    
    # 4. Final local search pass if time remains
    if time.time() < deadline - 60:
        points, lines, score = _local_search_points(points, cand_array, 27, max_iter=40, deadline=deadline)
    
    # 5. Final line refinement
    if time.time() < deadline - 10:
        lines, score = _refine_lines(points, cand_array, 27, deadline=deadline)
    
    # Safety check
    if len(lines) < 27:
        raise ValueError("Failed to generate valid 27 lines")
    
    return points, lines

# EVOLVE-BLOCK-END
