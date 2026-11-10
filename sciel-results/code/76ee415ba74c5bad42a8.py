import numpy as np
import time
import math
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidences(points, lines):
    """Count exact integer incidences between points and lines."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    total = 0
    for i in range(len(lns)):
        A, B, C = lns[i]
        val = A * pts[:, 0] + B * pts[:, 1] + C
        total += int(np.sum(val == 0))
    return total

def _is_valid(points, lines):
    """Validate points and lines per the task contract."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    if pts.shape != (54, 2) or lns.shape != (27, 3):
        return False
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    seen_lines = set()
    for l in lns:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm is None:
            return False
        if norm in seen_lines:
            return False
        seen_lines.add(norm)
    return True

def _find_optimal_lines(points, max_lines=27):
    """
    Find the set of up to max_lines lines with the highest total incidence count.
    """
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in line_map:
                line_map[norm] = set()
            line_map[norm].add(i)
            line_map[norm].add(j)
    
    line_data = []
    for norm, indices in line_map.items():
        cnt = len(indices)
        if cnt > 0:
            line_data.append((cnt, norm))
    
    line_data.sort(key=lambda x: x[0], reverse=True)
    
    selected = line_data[:max_lines]
    lines_arr = np.array([ln for _, ln in selected], dtype=np.int64)
    
    return lines_arr

def _build_grid(nx, ny):
    """Build a nx * ny grid of points. If nx*ny > 54, trim to 54."""
    xs = np.arange(nx)
    ys = np.arange(ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) > 54:
        points = points[:54]
    elif len(points) < 54:
        return None, None
    
    if len(points) != 54:
        return None, None
    
    lines = _find_optimal_lines(points, 27)
    return points, lines

def _local_search(points, lines, deadline):
    """
    Local search: iteratively try to improve by replacing the worst line
    with a better line through a pair of points not already covered.
    """
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    lines = np.array(lines, dtype=np.int64)
    current_score = _count_incidences(pts, lines)
    
    # Precompute all unique lines through point pairs
    all_lines = {}  # norm -> (A, B, C, set_of_point_indices)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in all_lines:
                all_lines[norm] = [A, B, C, set()]
            all_lines[norm][3].add(i)
            all_lines[norm][3].add(j)
    
    # Convert to list for faster access
    line_keys = list(all_lines.keys())
    line_data = {k: all_lines[k] for k in line_keys}
    
    # Track which lines are currently selected
    selected_norms = set()
    for l in lines:
        norm = _normalize_line(l[0], l[1], l[2])
        if norm is not None:
            selected_norms.add(norm)
    
    improved = True
    max_iterations = 5000
    
    while improved and time.time() < deadline and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Compute incidence count for each selected line
        line_counts = []
        for idx in range(len(lines)):
            A, B, C = lines[idx]
            val = A * pts[:, 0] + B * pts[:, 1] + C
            cnt = int(np.sum(val == 0))
            line_counts.append(cnt)
        
        # Find the line with minimum count
        min_idx = int(np.argmin(line_counts))
        min_count = line_counts[min_idx]
        min_norm = _normalize_line(lines[min_idx][0], lines[min_idx][1], lines[min_idx][2])
        
        # Try to find a better line to replace it
        best_gain = 0
        best_new_line = None
        
        for norm in line_keys:
            if norm in selected_norms:
                continue
            A, B, C, indices = line_data[norm]
            cnt = len(indices)
            gain = cnt - min_count
            if gain > best_gain:
                best_gain = gain
                best_new_line = (A, B, C)
        
        if best_gain > 0 and best_new_line is not None:
            # Replace the line
            lines[min_idx] = best_new_line
            selected_norms.discard(min_norm)
            selected_norms.add(_normalize_line(*best_new_line))
            current_score += best_gain
            improved = True
    
    return lines, current_score

def _try_point_swap(points, lines, deadline):
    """
    Try swapping individual points to improve the configuration.
    """
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    lines = np.array(lines, dtype=np.int64)
    current_score = _count_incidences(pts, lines)
    
    # For each point, try moving it to a nearby position
    for pi in range(n):
        if time.time() > deadline:
            break
        ox, oy = pts[pi]
        
        # Try nearby positions
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = ox + dx, oy + dy
                
                # Check if new position is already occupied
                if np.any((pts[:, 0] == nx_) & (pts[:, 1] == ny_)):
                    continue
                
                # Try the swap
                pts[pi] = [nx_, ny_]
                new_score = _count_incidences(pts, lines)
                
                if new_score > current_score:
                    current_score = new_score
                else:
                    pts[pi] = [ox, oy]
    
    return pts, current_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    best_pts = None
    best_lines = None
    best_score = -1
    
    # Try different grid aspect ratios
    configs = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
        (4, 14),
        (5, 11),
        (11, 5),
        (14, 4),
    ]
    
    for nx, ny in configs:
        if time.time() > deadline:
            break
        pts, lns = _build_grid(nx, ny)
        if pts is None:
            continue
        if not _is_valid(pts, lns):
            continue
        score = _count_incidences(pts, lns)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            best_lines = lns.copy()
    
    # If no grid worked, fallback to 6x9
    if best_pts is None:
        pts, lns = _build_grid(6, 9)
        if pts is not None:
            best_pts = pts
            best_lines = lns
    
    # If still nothing, return a simple valid construction
    if best_pts is None:
        xs = np.arange(1, 4)
        ys = np.arange(1, 19)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for a in range(1, 4):
            for b in range(1, 10):
                lines.append((a, -1, b))
        best_lines = np.array(lines, dtype=np.int64)
    
    # Local search to improve the best configuration
    if time.time() < deadline - 60:
        improved_lines, improved_score = _local_search(
            best_pts, best_lines, deadline - 60
        )
        if improved_score > best_score:
            best_lines = improved_lines
            best_score = improved_score
    
    # Try point swaps if time remains
    if time.time() < deadline - 30:
        swapped_pts, swapped_score = _try_point_swap(
            best_pts, best_lines, deadline - 30
        )
        if swapped_score > best_score:
            best_pts = swapped_pts
            best_score = swapped_score
            # Re-run line optimization after point changes
            if time.time() < deadline - 10:
                new_lines = _find_optimal_lines(best_pts, 27)
                new_score = _count_incidences(best_pts, new_lines)
                if new_score > best_score:
                    best_lines = new_lines
                    best_score = new_score
    
    return best_pts, best_lines
