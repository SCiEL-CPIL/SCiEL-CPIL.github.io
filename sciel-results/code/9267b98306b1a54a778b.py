# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients so first nonzero is positive and gcd divides all."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact incidences between points and lines using int64 arithmetic."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _find_best_lines_for_points(points, num_lines):
    """Find the best num_lines lines that maximize incidences with given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _normalize_line(points[j][1] - points[i][1], points[i][0] - points[j][0], points[j][0]*points[i][1] - points[i][0]*points[j][1])
            if line is not None:
                line_counts[line] = line_counts.get(line, 0) + 1
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    selected = []
    selected_set = set()
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line not in selected_set:
            selected.append(line)
            selected_set.add(line)
    if len(selected) < num_lines:
        existing_norms = set(selected)
        counter = 100
        while len(selected) < num_lines:
            test_line = _normalize_line(1, 1, -counter)
            if test_line not in existing_norms:
                selected.append(test_line)
                existing_norms.add(test_line)
            counter += 1
    return np.array(selected, dtype=np.int64)

def _joint_local_search(points, lines, deadline):
    """
    Joint local search: perturb point positions AND line coefficients.
    This allows co-evolution of both elements.
    """
    pts = points.copy().astype(np.int64)
    lns = lines.copy().astype(np.int64)
    n_pts = len(pts)
    n_lns = len(lns)
    
    # Track occupied points for O(1) lookup
    occupied = set(map(tuple, pts))
    
    def _total_inc():
        total = 0
        for i in range(n_lns):
            A, B, C = lns[i]
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            total += np.sum(vals == 0)
        return total
    
    current_inc = _total_inc()
    start_time = time.time()
    no_improve_rounds = 0
    max_no_improve = 50
    radius = 2
    
    while (time.time() - start_time) < deadline - 15 and no_improve_rounds < max_no_improve:
        improved = False
        no_improve_rounds += 1
        
        # --- Phase 1: Try moving individual points ---
        for i in range(n_pts):
            if (time.time() - start_time) > deadline - 15:
                return pts, lns, current_inc
            
            old_x, old_y = int(pts[i, 0]), int(pts[i, 1])
            old_count = 0
            for j in range(n_lns):
                A, B, C = lns[j]
                if A * old_x + B * old_y + C == 0:
                    old_count += 1
            
            best_new_x, best_new_y = old_x, old_y
            best_new_count = old_count
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    if (new_x, new_y) in occupied:
                        continue
                    
                    new_count = 0
                    for j in range(n_lns):
                        A, B, C = lns[j]
                        if A * new_x + B * new_y + C == 0:
                            new_count += 1
                    
                    if new_count > best_new_count:
                        best_new_count = new_count
                        best_new_x, best_new_y = new_x, new_y
            
            if best_new_count > old_count:
                occupied.discard((old_x, old_y))
                pts[i, 0] = best_new_x
                pts[i, 1] = best_new_y
                occupied.add((best_new_x, best_new_y))
                current_inc += best_new_count - old_count
                improved = True
        
        # --- Phase 2: Try perturbing individual lines ---
        for j in range(n_lns):
            if (time.time() - start_time) > deadline - 15:
                return pts, lns, current_inc
            
            old_A, old_B, old_C = int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2])
            
            # Count current incidence for this line
            old_line_inc = 0
            vals = old_A * pts[:, 0] + old_B * pts[:, 1] + old_C
            old_line_inc = np.sum(vals == 0)
            
            best_new_A, best_new_B, best_new_C = old_A, old_B, old_C
            best_new_line_inc = old_line_inc
            
            # Try perturbing A, B, C by small amounts
            for dA in range(-radius, radius + 1):
                for dB in range(-radius, radius + 1):
                    for dC in range(-radius, radius + 1):
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = old_A + dA, old_B + dB, old_C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        
                        # Normalize to check for duplicates
                        norm_new = _normalize_line(new_A, new_B, new_C)
                        if norm_new is None:
                            continue
                        
                        # Check if this line already exists in our set
                        is_dup = False
                        for k in range(n_lns):
                            if k == j:
                                continue
                            norm_k = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                            if norm_new == norm_k:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        # Count new incidence
                        vals_new = new_A * pts[:, 0] + new_B * pts[:, 1] + new_C
                        new_line_inc = np.sum(vals_new == 0)
                        
                        if new_line_inc > best_new_line_inc:
                            best_new_line_inc = new_line_inc
                            best_new_A, best_new_B, best_new_C = new_A, new_B, new_C
            
            if best_new_line_inc > old_line_inc:
                lns[j, 0] = best_new_A
                lns[j, 1] = best_new_B
                lns[j, 2] = best_new_C
                current_inc += best_new_line_inc - old_line_inc
                improved = True
        
        if not improved:
            # Try expanding radius slightly
            if radius < 5:
                radius += 1
            else:
                break
    
    return pts, lns, current_inc

def _validate_and_fix(points, lines):
    """Validate points and lines, fix any issues."""
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        return None
    
    # Ensure lines are valid (no A=B=0)
    valid_mask = ~((lines[:, 0] == 0) & (lines[:, 1] == 0))
    if not np.all(valid_mask):
        return None
    
    # Check for duplicate lines
    normalized_lines = [_normalize_line(int(l[0]), int(l[1]), int(l[2])) for l in lines]
    seen = set()
    for nl in normalized_lines:
        if nl is None or nl in seen:
            return None
        seen.add(nl)
    
    return True

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    Uses joint local search to co-evolve points and lines.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    # Try multiple initial configurations
    configs = [
        # 6x9 grid
        (np.arange(1, 7), np.arange(1, 10)),
        # 9x6 grid (transposed)
        (np.arange(1, 10), np.arange(1, 7)),
        # 3x18 grid
        (np.arange(1, 4), np.arange(1, 19)),
        # 18x3 grid
        (np.arange(1, 19), np.arange(1, 4)),
    ]
    
    for xs, ys in configs:
        if (time.time() - start_time) > deadline - 60:
            break
        
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        if len(points) != 54:
            continue
        
        lines = _find_best_lines_for_points(points, 27)
        
        # Run joint local search with time budget
        time_budget = min(120, max(30, (deadline - 30) - (time.time() - start_time)))
        pts, lns, inc = _joint_local_search(points, lines, start_time + time_budget)
        
        if _validate_and_fix(pts, lns):
            if inc > best_inc:
                best_inc = inc
                best_points = pts.copy()
                best_lines = lns.copy()
    
    # If no valid result found, fallback to first config
    if best_points is None:
        xs, ys = configs[0]
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        best_lines = _find_best_lines_for_points(best_points, 27)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
