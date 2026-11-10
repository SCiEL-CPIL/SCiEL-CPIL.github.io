# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _gcd3(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    g = _gcd3(A, B, C)
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    count = 0
    for i in range(lines.shape[0]):
        A, B, C = int(lines[i, 0]), int(lines[i, 1]), int(lines[i, 2])
        if A == 0 and B == 0:
            continue
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += int(np.sum(vals == 0))
    return count

def _validate(points, lines):
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

def _generate_candidate_lines(points, max_slope=6, max_offset_range=30):
    """Generate all candidate lines that pass through at least 2 points."""
    n = points.shape[0]
    lines_set = set()
    # For each pair of points, compute the line through them
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = int(points[i, 0]), int(points[i, 1])
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            lines_set.add(norm)
    
    # Also add axis-aligned and simple slope lines
    xs = np.unique(points[:, 0])
    ys = np.unique(points[:, 1])
    for x in xs:
        lines_set.add(_normalize_line(1, 0, -x))
    for y in ys:
        lines_set.add(_normalize_line(0, 1, -y))
    
    # Add lines with small integer slopes
    for slope in range(-max_slope, max_slope + 1):
        if slope == 0:
            continue
        for c in range(-max_offset_range, max_offset_range + 1):
            norm = _normalize_line(slope, -1, c)
            lines_set.add(norm)
    
    # Add lines with small rational slopes (p/q form)
    for p in range(-max_slope, max_slope + 1):
        for q in range(1, max_slope + 1):
            if p == 0:
                continue
            for c in range(-max_offset_range, max_offset_range + 1):
                norm = _normalize_line(p, -q, c)
                lines_set.add(norm)
    
    return list(lines_set)

def _local_search(points, lines, max_iterations=500, time_limit=None):
    """Local search: try swapping individual lines for better ones."""
    start = time.time()
    best_lines = lines.copy()
    best_score = _count_incidents(points, best_lines)
    
    # Precompute incidence counts for all candidate lines
    candidates = _generate_candidate_lines(points, max_slope=6, max_offset_range=30)
    
    # Score each candidate line
    line_scores = {}
    for line in candidates:
        A, B, C = line
        if A == 0 and B == 0:
            continue
        vals = A * points[:, 0] + B * points[:, 1] + C
        score = int(np.sum(vals == 0))
        line_scores[line] = score
    
    # Current line set
    current_norms = set()
    for i in range(best_lines.shape[0]):
        A, B, C = int(best_lines[i, 0]), int(best_lines[i, 1]), int(best_lines[i, 2])
        current_norms.add(_normalize_line(A, B, C))
    
    # Sort candidates by score descending
    sorted_candidates = sorted(line_scores.items(), key=lambda x: x[1], reverse=True)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        if time_limit and time.time() > time_limit:
            break
        improved = False
        iteration += 1
        
        # Try swapping each current line with a better candidate
        for i in range(best_lines.shape[0]):
            A, B, C = int(best_lines[i, 0]), int(best_lines[i, 1]), int(best_lines[i, 2])
            current_norm = _normalize_line(A, B, C)
            current_score = line_scores.get(current_norm, 0)
            
            # Find best candidate not in current set
            for cand, cand_score in sorted_candidates:
                if cand in current_norms:
                    continue
                if cand_score > current_score:
                    # Swap: remove current_norm, add cand
                    new_norms = set(current_norms)
                    new_norms.discard(current_norm)
                    new_norms.add(cand)
                    
                    if len(new_norms) == 27:
                        new_lines = best_lines.copy()
                        new_lines[i] = np.array(cand, dtype=np.int64)
                        
                        new_score = _count_incidents(points, new_lines)
                        if new_score > best_score:
                            best_lines = new_lines
                            best_score = new_score
                            current_norms = new_norms
                            improved = True
                            break
            if improved:
                break
    
    return best_lines, best_score

def _perturb_points(points, n_perturb=10, time_limit=None):
    """Perturb a few points to escape local optima, then re-optimize lines."""
    n = points.shape[0]
    best_points = points.copy()
    best_score = _count_incidents(points, best_points)
    
    for _ in range(n_perturb):
        if time_limit and time.time() > time_limit:
            break
        
        new_points = best_points.copy()
        n_move = np.random.randint(1, 4)
        indices = np.random.choice(n, n_move, replace=False)
        
        for idx in indices:
            base_x = int(new_points[idx, 0])
            base_y = int(new_points[idx, 1])
            moved = False
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = base_x + dx
                    new_y = base_y + dy
                    if np.any((new_points[:, 0] == new_x) & (new_points[:, 1] == new_y)):
                        continue
                    new_points[idx, 0] = new_x
                    new_points[idx, 1] = new_y
                    moved = True
                    break
                if moved:
                    break
        
        if not moved:
            continue
        
        candidates = _generate_candidate_lines(new_points, max_slope=4, max_offset_range=20)
        line_scores = {}
        for line in candidates:
            A, B, C = line
            vals = A * new_points[:, 0] + B * new_points[:, 1] + C
            score = int(np.sum(vals == 0))
            line_scores[line] = score
        
        sorted_cands = sorted(line_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        selected_norms = set()
        for cand, score in sorted_cands:
            if len(selected) >= 27:
                break
            if cand not in selected_norms:
                selected.append(cand)
                selected_norms.add(cand)
        
        while len(selected) < 27:
            selected.append((1, 2, 1))
        
        new_lines = np.array(selected[:27], dtype=np.int64)
        new_score = _count_incidents(new_points, new_lines)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
    
    return best_points, best_score

def _build_grid_and_lines(gx, gy, n_diag=12):
    """Build a gx x gy grid with all vertical/horizontal lines and best diagonal lines."""
    xs = np.arange(1, gx + 1)
    ys = np.arange(1, gy + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    for i in range(1, gx + 1):
        lines.append((1, 0, -i))
    
    for j in range(1, gy + 1):
        lines.append((0, 1, -j))
    
    all_diag = []
    
    for c in range(-(gy - 1), gx):
        count = 0
        for x in range(1, gx + 1):
            y = x + c
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            all_diag.append((count, (1, -1, c)))
    
    for c in range(2, gx + gy + 1):
        count = 0
        for x in range(1, gx + 1):
            y = c - x
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            all_diag.append((count, (1, 1, -c)))
    
    all_diag.sort(key=lambda x: x[0], reverse=True)
    
    for count, line in all_diag[:n_diag]:
        lines.append(line)
    
    while len(lines) < 27:
        lines.append((1, 2, 1))
    
    lines = lines[:27]
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines maximizing incidences.
    Uses grid-based construction with local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    best_incidents = 0
    best_points = None
    best_lines = None
    
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
        if time.time() > deadline:
            break
        
        n_lines_axis = gx + gy
        n_diag = 27 - n_lines_axis
        if n_diag < 0:
            continue
        
        points, lines = _build_grid_and_lines(gx, gy, n_diag)
        
        if not _validate(points, lines):
            continue
        
        incidents = _count_incidents(points, lines)
        
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            lines, incidents = _local_search(
                points, lines, 
                max_iterations=300,
                time_limit=start_time + remaining_time - 5
            )
        
        remaining_time = deadline - time.time()
        if remaining_time > 20:
            new_points, new_incidents = _perturb_points(
                points, n_perturb=10,
                time_limit=start_time + remaining_time - 10
            )
            
            if new_incidents > incidents:
                candidates = _generate_candidate_lines(new_points, max_slope=4, max_offset_range=20)
                line_scores = {}
                for line in candidates:
                    A, B, C = line
                    vals = A * new_points[:, 0] + B * new_points[:, 1] + C
                    score = int(np.sum(vals == 0))
                    line_scores[line] = score
                
                sorted_cands = sorted(line_scores.items(), key=lambda x: x[1], reverse=True)
                selected = []
                selected_norms = set()
                for cand, score in sorted_cands:
                    if len(selected) >= 27:
                        break
                    if cand not in selected_norms:
                        selected.append(cand)
                        selected_norms.add(cand)
                
                while len(selected) < 27:
                    selected.append((1, 2, 1))
                
                new_lines = np.array(selected[:27], dtype=np.int64)
                
                remaining_time = deadline - time.time()
                if remaining_time > 5:
                    new_lines, new_incidents = _local_search(
                        new_points, new_lines,
                        max_iterations=200,
                        time_limit=start_time + remaining_time - 2
                    )
                
                points = new_points
                lines = new_lines
                incidents = new_incidents
        
        if incidents > best_incidents:
            best_incidents = incidents
            best_points = points.copy()
            best_lines = lines.copy()
    
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines = []
        for i in range(1, 7):
            lines.append((1, 0, -i))
        for j in range(1, 10):
            lines.append((0, 1, -j))
        c_vals_slope_1 = [0, 1, 2, 3, -1, 4]
        for c in c_vals_slope_1:
            lines.append((1, -1, c))
        c_vals_slope_neg_1 = [7, 8, 9, 10, 6, 11]
        for c in c_vals_slope_neg_1:
            lines.append((1, 1, -c))
        best_lines = np.array(lines, dtype=np.int64)
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
