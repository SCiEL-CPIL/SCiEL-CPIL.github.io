import numpy as np
import time
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences exactly."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = lines[:, 0:1].astype(np.int64).T
    B = lines[:, 1:2].astype(np.int64).T
    C = lines[:, 2:3].astype(np.int64).T
    total = px * A + py * B + C
    return int(np.sum(total == 0))

def _generate_grid_points(w, h):
    """Generate a w x h grid of points starting from (1,1)."""
    xs = np.arange(1, w + 1)
    ys = np.arange(1, h + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _generate_candidate_lines(points, max_coord=20):
    """
    Generate all unique lines that pass through at least 2 points of the grid,
    or lines with small integer coefficients within a reasonable range.
    To be safe and fast, we generate all lines (A,B,C) with |A|,|B|,|C| <= max_coord 
    that are normalized, and filter those that intersect the bounding box of points.
    """
    candidates = set()
    min_x = int(np.min(points[:, 0]))
    max_x = int(np.max(points[:, 0]))
    min_y = int(np.min(points[:, 1]))
    max_y = int(np.max(points[:, 1]))
    
    # We want lines that actually pass through the point set or are close.
    # A line A*x + B*y + C = 0 intersects the box [min_x, max_x] x [min_y, max_y]
    # if there exists (x,y) in the box such that A*x + B*y + C = 0.
    # This is equivalent to: -C is in the range of A*x + B*y for x in [min_x, max_x], y in [min_y, max_y].
    
    for A in range(-max_coord, max_coord + 1):
        for B in range(-max_coord, max_coord + 1):
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), abs(B))
            if g > 0:
                A_norm, B_norm = A // g, B // g
            else:
                A_norm, B_norm = A, B
            if A_norm < 0 or (A_norm == 0 and B_norm < 0):
                A_norm, B_norm = -A_norm, -B_norm
                
            # Determine valid range for C
            # C = -(A*x + B*y)
            # Min and max of A*x + B*y over the box
            # Since A, B can be negative, we check corners
            vals = [
                A_norm * min_x + B_norm * min_y,
                A_norm * min_x + B_norm * max_y,
                A_norm * max_x + B_norm * min_y,
                A_norm * max_x + B_norm * max_y
            ]
            min_sum = min(vals)
            max_sum = max(vals)
            
            # C must be such that -C is between min_sum and max_sum
            # So C is between -max_sum and -min_sum
            for C in range(-max_sum, -min_sum + 1):
                candidates.add(_normalize_line(A_norm, B_norm, C))
                
    return list(candidates)

def _select_best_lines(points, candidates, k=27):
    """
    Select the k lines from candidates that maximize total incidences with points.
    Since the score is the sum of incidences for each line, and incidences of different lines
    do not interfere (we count pairs (p,l)), the optimal strategy is to simply pick the
    k lines with the highest individual incidence counts.
    """
    if len(candidates) == 0:
        return np.zeros((k, 3), dtype=np.int64)
    
    cand_arr = np.array(candidates, dtype=np.int64)
    px = points[:, 0:1].astype(np.int64)
    py = points[:, 1:2].astype(np.int64)
    A = cand_arr[:, 0:1].astype(np.int64).T
    B = cand_arr[:, 1:2].astype(np.int64).T
    C = cand_arr[:, 2:3].astype(np.int64).T
    
    total = px * A + py * B + C
    inc_counts = np.sum(total == 0, axis=0)
    
    # Get indices of top k
    top_indices = np.argsort(inc_counts)[::-1][:k]
    return cand_arr[top_indices]

def _evaluate_config(points, lines):
    return _incidence_count(points, lines)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Try several grid configurations
    # 54 points can be 6x9, 9x6, 3x18, 18x3, 2x27, 27x2, 1x54, 54x1
    # Also non-rectangular subsets? For now, stick to rectangles.
    # The "inspiration" mentions 2.17 score. Let's see what grid yields that.
    # A 6x9 grid has 54 points.
    # Lines: 6 vertical, 9 horizontal.
    # Diagonals and other slopes add more.
    
    grid_shapes = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    
    # We can also try "distorted" grids or other sets, but rectangles are strong baselines.
    # Let's also try a 7x8 grid minus 2 points? No, let's stick to full rectangles first.
    # Maybe 54 points in a 10x10 grid (sparse)?
    # The prompt says "54 points". It doesn't say they must form a grid.
    # However, grids are highly symmetric and likely optimal or near-optimal for integer coordinates.
    
    for w, h in grid_shapes:
        if time.time() > deadline - 10:
            break
            
        points = _generate_grid_points(w, h)
        
        # Generate candidates
        # max_coord should be large enough to capture all relevant lines.
        # For a w x h grid, slopes can be up to h/w or w/h.
        # Intercepts can be large.
        # Let's use max_coord = max(w, h) * 2 + 10 to be safe.
        max_coord = max(w, h) * 2 + 10
        candidates = _generate_candidate_lines(points, max_coord)
        
        # Select best 27 lines
        lines = _select_best_lines(points, candidates, 27)
        
        score = _evaluate_config(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
            
    # If we have time, try local search on the best configuration found
    if best_points is not None and time.time() < deadline - 20:
        # Simple local search: try moving each point slightly and re-selecting lines
        # This is expensive if we re-generate candidates each time.
        # Instead, just re-select lines from the existing candidate pool?
        # The candidate pool was generated for the original grid.
        # If we move a point, the "best" lines might change, but the pool might still contain them.
        # Let's just do a greedy line re-selection for the fixed points first (already done).
        # Now try to improve points.
        
        current_points = best_points.copy()
        current_lines = best_lines.copy()
        current_score = best_score
        
        # Precompute candidate array for fast incidence check
        # We need a larger candidate pool to allow point moves to be beneficial.
        # Let's generate a larger pool based on the bounding box of current points.
        min_x = int(np.min(current_points[:, 0])) - 5
        max_x = int(np.max(current_points[:, 0])) + 5
        min_y = int(np.min(current_points[:, 1])) - 5
        max_y = int(np.max(current_points[:, 1])) + 5
        
        # Re-generate candidates for the expanded box
        max_coord = max(max_x - min_x, max_y - min_y) * 2 + 10
        # Note: _generate_candidate_lines uses points to determine box.
        # We can pass a dummy points array or modify it.
        # Let's just create a dummy grid covering the expanded box to generate candidates.
        # Actually, _generate_candidate_lines takes points to get min/max.
        # So we can just pass the current points, but it uses their min/max.
        # To expand, we can create a "virtual" points set or just increase max_coord.
        # Increasing max_coord is safer.
        
        candidates = _generate_candidate_lines(current_points, max_coord=max_coord)
        cand_arr = np.array(candidates, dtype=np.int64)
        
        # Local search on points
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current_points)):
                if time.time() > deadline - 5:
                    break
                x, y = current_points[i]
                # Try small moves
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        # Check duplicate
                        is_dup = False
                        for j in range(len(current_points)):
                            if i == j: continue
                            if current_points[j][0] == nx and current_points[j][1] == ny:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                            
                        new_points = current_points.copy()
                        new_points[i] = [nx, ny]
                        
                        # Re-select best lines for new points
                        new_lines = _select_best_lines(new_points, candidates, 27)
                        new_score = _evaluate_config(new_points, new_lines)
                        
                        if new_score > current_score:
                            current_points = new_points
                            current_lines = new_lines
                            current_score = new_score
                            improved = True
                            # Update best
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points.copy()
                                best_lines = current_lines.copy()
                            break # Restart inner loops? Or just continue?
                    if improved:
                        break
                if improved:
                    break
    
    # Final validation and return
    if best_points is None:
        # Fallback
        best_points = _generate_grid_points(6, 9)
        candidates = _generate_candidate_lines(best_points, 20)
        best_lines = _select_best_lines(best_points, candidates, 27)
        
    return best_points, best_lines
