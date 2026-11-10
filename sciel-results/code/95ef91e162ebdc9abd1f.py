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
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return np.sum(total == 0)

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

def _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=4):
    """Generate candidate lines that intersect the bounding box."""
    candidates = []
    for A in range(-max_coef, max_coef + 1):
        for B in range(-max_coef, max_coef + 1):
            if A == 0 and B == 0:
                continue
            # Determine C range
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

def _local_search_points(points, cand_array, k=27, max_iter=100, deadline=None):
    """
    Local search: try moving individual points to improve total incidence.
    """
    best_points = points.copy()
    best_score = _incidence_count(best_points, _select_best_lines(best_points, cand_array, k))
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        improved = False
        n_points = len(best_points)
        
        for i in range(n_points):
            old_x, old_y = best_points[i, 0], best_points[i, 1]
            
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Skip if this creates a duplicate point
                    if np.any((best_points[:, 0] == new_x) & (best_points[:, 1] == new_y)):
                        continue
                    
                    # Temporarily move point
                    best_points[i, 0], best_points[i, 1] = new_x, new_y
                    new_score = _incidence_count(best_points, _select_best_lines(best_points, cand_array, k))
                    
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        # Revert
                        best_points[i, 0], best_points[i, 1] = old_x, old_y
        
        if not improved:
            break
    
    return best_points, best_score

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses a 7x8 grid with 2 removed corners and local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # 1. Define Points: 7x8 grid with 2 corners removed
    # This gives 56 - 2 = 54 points
    xs = np.arange(1, 8)  # 1 to 7
    ys = np.arange(1, 9)  # 1 to 8
    px, py = np.meshgrid(xs, ys, indexing="ij")
    
    # Remove 2 corners to get 54 points
    # Remove (1,1) and (7,8) - opposite corners
    mask = ~(((px == 1) & (py == 1)) | ((px == 7) & (py == 8)))
    points = np.stack([px[mask], py[mask]], axis=1).astype(np.int64)
    
    # 2. Generate Candidate Lines
    x_min, x_max = 1, 7
    y_min, y_max = 1, 8
    cand_array = _generate_candidate_lines(x_min, x_max, y_min, y_max, max_coef=4)
    
    # 3. Select Initial Best Lines
    lines = _select_best_lines(points, cand_array, 27)
    initial_score = _incidence_count(points, lines)
    
    # 4. Local Search to Improve Points
    if time.time() < deadline - 60:
        points, final_score = _local_search_points(points, cand_array, 27, max_iter=50, deadline=deadline)
        lines = _select_best_lines(points, cand_array, 27)
        final_score = _incidence_count(points, lines)
    
    # Safety check
    if len(lines) < 27:
        raise ValueError("Failed to generate valid 27 lines")
    
    return points, lines

# EVOLVE-BLOCK-END
