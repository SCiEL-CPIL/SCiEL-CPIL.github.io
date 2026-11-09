# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _gcd3(a, b, c):
    """Compute gcd of three non-negative integers."""
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    """Normalize a line (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = _gcd3(A, B, C)
    if g > 0:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    """Count point-line incidences exactly using integer arithmetic."""
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            continue
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += int(np.sum(vals == 0))
    return count

def _validate(points, lines):
    """Check if points and lines satisfy all constraints. Return True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    
    if not np.all(np.isfinite(points.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lines.astype(np.float64))):
        return False
    
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    
    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] != 54:
        return False
    
    seen_lines = set()
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    
    return True

def _build_grid_points(gx, gy):
    """Build a gx x gy grid of points."""
    xs = np.arange(1, gx + 1)
    ys = np.arange(1, gy + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _compute_line_incidence(points, A, B, C):
    """Compute the number of points on line Ax + By + C = 0."""
    if A == 0 and B == 0:
        return 0
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _generate_candidate_lines(points, max_candidates=200):
    """Generate candidate lines that pass through at least 2 points."""
    n = points.shape[0]
    candidates = {}
    
    # For each pair of points, compute the line through them
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            # Line through (x1,y1) and (x2,y2): (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            norm = _normalize_line(A, B, C)
            if norm not in candidates:
                candidates[norm] = _compute_line_incidence(points, *norm)
    
    # Also add lines with small coefficients that might have high incidence
    # Vertical lines
    for x in range(-5, 60):
        norm = _normalize_line(1, 0, -x)
        if norm not in candidates:
            candidates[norm] = _compute_line_incidence(points, *norm)
    # Horizontal lines
    for y in range(-5, 60):
        norm = _normalize_line(0, 1, -y)
        if norm not in candidates:
            candidates[norm] = _compute_line_incidence(points, *norm)
    # Diagonal lines with small slopes
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            for C in range(-50, 51):
                norm = _normalize_line(A, B, C)
                if norm not in candidates:
                    candidates[norm] = _compute_line_incidence(points, *norm)
    
    # Sort by incidence count descending
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:max_candidates]

def _local_search(points, lines, deadline, max_iterations=5000):
    """Perform local search to improve line selection."""
    n_lines = lines.shape[0]
    current_incidents = _count_incidents(points, lines)
    
    # Precompute incidence for all current lines
    line_incs = []
    for i in range(n_lines):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        line_incs.append(_compute_line_incidence(points, A, B, C))
    
    # Generate candidate lines
    candidates = _generate_candidate_lines(points, max_candidates=300)
    candidate_lines = [c[0] for c in candidates]
    candidate_incs = [c[1] for c in candidates]
    
    # Build a map from normalized line to index in candidate list
    candidate_map = {}
    for idx, line in enumerate(candidate_lines):
        candidate_map[line] = idx
    
    best_incidents = current_incidents
    best_lines = lines.copy()
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and time.time() < deadline:
        improved = False
        iteration += 1
        
        # Try swapping each line with a better candidate
        for i in range(n_lines):
            if time.time() > deadline:
                break
            
            current_line = tuple(int(x) for x in lines[i])
            current_norm = _normalize_line(*current_line)
            current_inc = line_incs[i]
            
            # Find the best candidate to replace this line
            best_j = -1
            best_gain = 0
            for j in range(len(candidate_lines)):
                cand_line = candidate_lines[j]
                if cand_line == current_norm:
                    continue
                # Check if this candidate is already used
                already_used = False
                for k in range(n_lines):
                    if k == i:
                        continue
                    if _normalize_line(*tuple(int(x) for x in lines[k])) == cand_line:
                        already_used = True
                        break
                if already_used:
                    continue
                
                gain = candidate_incs[j] - current_inc
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j >= 0 and best_gain > 0:
                new_line = np.array(candidate_lines[best_j], dtype=np.int64)
                lines[i] = new_line
                line_incs[i] = candidate_incs[best_j]
                current_incidents += best_gain
                improved = True
                
                if current_incidents > best_incidents:
                    best_incidents = current_incidents
                    best_lines = lines.copy()
    
    return best_incidents, best_lines

def _try_grid_configuration(gx, gy, deadline):
    """Try a grid configuration with local search."""
    if gx * gy != 54:
        return None
    
    points = _build_grid_points(gx, gy)
    
    # Initial lines: all vertical, all horizontal, and best diagonals
    lines_list = []
    
    # Vertical lines
    for i in range(1, gx + 1):
        lines_list.append((1, 0, -i))
    
    # Horizontal lines
    for j in range(1, gy + 1):
        lines_list.append((0, 1, -j))
    
    # Add diagonal lines to reach 27 total
    n_needed = 27 - len(lines_list)
    
    # Generate diagonal candidates
    diag_candidates = []
    for c in range(-(gy - 1), gx + 1):
        # Slope 1: y - x = c -> (1, -1, c)
        inc = _compute_line_incidence(points, 1, -1, c)
        if inc > 0:
            diag_candidates.append((inc, (1, -1, c)))
    for c in range(2, gx + gy + 1):
        # Slope -1: x + y = c -> (1, 1, -c)
        inc = _compute_line_incidence(points, 1, 1, -c)
        if inc > 0:
            diag_candidates.append((inc, (1, 1, -c)))
    
    diag_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for inc, line in diag_candidates[:n_needed]:
        lines_list.append(line)
    
    # Pad if needed
    while len(lines_list) < 27:
        lines_list.append((1, 2, 1))
    
    lines_list = lines_list[:27]
    lines = np.array(lines_list, dtype=np.int64)
    
    if not _validate(points, lines):
        return None
    
    # Run local search
    incidents, improved_lines = _local_search(points, lines, deadline)
    
    if not _validate(points, improved_lines):
        return None
    
    return incidents, points, improved_lines

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    best_incidents = 0
    best_points = None
    best_lines = None
    
    # Grid configurations where gx * gy = 54
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    for gx, gy in configs:
        if time.time() > deadline - 10:
            break
        
        result = _try_grid_configuration(gx, gy, deadline)
        if result is not None:
            incidents, points, lines = result
            if incidents > best_incidents:
                best_incidents = incidents
                best_points = points.copy()
                best_lines = lines.copy()
    
    # If no valid configuration found, use fallback
    if best_points is None:
        points = _build_grid_points(6, 9)
        lines_list = []
        for i in range(1, 7):
            lines_list.append((1, 0, -i))
        for j in range(1, 10):
            lines_list.append((0, 1, -j))
        for c in [0, 1, 2, 3, -1, 4]:
            lines_list.append((1, -1, c))
        for c in [7, 8, 9, 10, 6, 11]:
            lines_list.append((1, 1, -c))
        best_lines = np.array(lines_list[:27], dtype=np.int64)
        best_points = points
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
