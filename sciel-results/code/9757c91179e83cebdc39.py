import numpy as np
import time
import math
from math import gcd

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

def _generate_lines_for_points(pts):
    """Generate all unique lines determined by pairs of points, with incidence counts."""
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
    return line_data

def _select_top_lines(line_data, k):
    """Select top k lines by incidence count."""
    selected = line_data[:k]
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
    
    line_data = _generate_lines_for_points(points)
    lines = _select_top_lines(line_data, 27)
    return points, lines

def _local_search_points(points, lines, deadline, max_iters=500):
    """
    Local search: try moving individual points to nearby integer positions
    to increase total incidence count.
    """
    pts = points.copy().astype(np.int64)
    lns = lines.copy().astype(np.int64)
    best_score = _count_incidences(pts, lns)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        # Try moving each point in 8 directions
        for i in range(54):
            if time.time() > deadline:
                break
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[i, 0] += dx
                    new_pts[i, 1] += dy
                    
                    # Check for duplicate points
                    dup = False
                    for j in range(54):
                        if j == i:
                            continue
                        if new_pts[j, 0] == new_pts[i, 0] and new_pts[j, 1] == new_pts[i, 1]:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    score = _count_incidences(new_pts, lns)
                    if score > best_score:
                        pts = new_pts
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try line replacement: for each line, try replacing with a better line
            # that has more incidences with current points
            for li in range(27):
                if time.time() > deadline:
                    break
                # Generate all lines through pairs of points
                line_data = _generate_lines_for_points(pts)
                # Find best line not already in our set
                current_norms = set()
                for l in lns:
                    n = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
                    if n:
                        current_norms.add(n)
                
                best_new_line = None
                best_new_cnt = 0
                for cnt, norm in line_data:
                    if norm not in current_norms and cnt > 0:
                        if cnt > best_new_cnt:
                            best_new_cnt = cnt
                            best_new_line = norm
                
                if best_new_line is not None:
                    # Check if replacing line li with best_new_line improves score
                    old_A, old_B, old_C = lns[li]
                    old_val = old_A * pts[:, 0] + old_B * pts[:, 1] + old_C
                    old_cnt = int(np.sum(old_val == 0))
                    
                    new_A, new_B, new_C = best_new_line
                    new_val = new_A * pts[:, 0] + new_B * pts[:, 1] + new_C
                    new_cnt = int(np.sum(new_val == 0))
                    
                    if new_cnt > old_cnt:
                        lns[li] = [new_A, new_B, new_C]
                        best_score += (new_cnt - old_cnt)
                        improved = True
                        break
            
            if not improved:
                break
    
    return pts, lns, best_score

def _try_perturbed_grid(base_pts, deadline):
    """Try perturbing grid points slightly and re-optimizing lines."""
    pts = base_pts.copy().astype(np.int64)
    n = len(pts)
    
    # Generate all lines for base grid
    line_data = _generate_lines_for_points(pts)
    lines = _select_top_lines(line_data, 27)
    base_score = _count_incidences(pts, lines)
    
    best_pts = pts.copy()
    best_lines = lines.copy()
    best_score = base_score
    
    # Try moving individual points
    for i in range(n):
        if time.time() > deadline:
            break
        for dx in [-2, -1, 0, 1, 2]:
            if time.time() > deadline:
                break
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                
                # Check duplicates
                dup = False
                for j in range(n):
                    if j != i and new_pts[j, 0] == new_pts[i, 0] and new_pts[j, 1] == new_pts[i, 1]:
                        dup = True
                        break
                if dup:
                    continue
                
                # Re-optimize lines for this point configuration
                ld = _generate_lines_for_points(new_pts)
                nl = _select_top_lines(ld, 27)
                score = _count_incidences(new_pts, nl)
                
                if score > best_score:
                    best_score = score
                    best_pts = new_pts
                    best_lines = nl
    
    return best_pts, best_lines, best_score

def szemeredi_trotter_incidences():
    deadline = time.time() + 590.0
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
    
    # If no grid worked, fallback
    if best_pts is None:
        pts, lns = _build_grid(6, 9)
        if pts is not None:
            best_pts = pts
            best_lines = lns
            best_score = _count_incidences(pts, lns)
    
    # Local search to improve
    if best_pts is not None and time.time() < deadline - 30:
        pts, lns, score = _local_search_points(best_pts, best_lines, deadline, max_iters=200)
        if score > best_score:
            best_score = score
            best_pts = pts
            best_lines = lns
    
    # Try perturbed grids
    if best_pts is not None and time.time() < deadline - 20:
        for nx, ny in [(6, 9), (9, 6), (3, 18)]:
            if time.time() > deadline - 10:
                break
            base_pts, _ = _build_grid(nx, ny)
            if base_pts is None:
                continue
            ppts, plns, pscore = _try_perturbed_grid(base_pts, deadline)
            if pscore > best_score:
                best_score = pscore
                best_pts = ppts
                best_lines = plns
    
    # Final validation
    if best_pts is not None and _is_valid(best_pts, best_lines):
        return best_pts, best_lines
    
    # Fallback: simple valid construction
    xs = np.arange(1, 4)
    ys = np.arange(1, 19)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines = []
    for a in range(1, 4):
        for b in range(1, 10):
            lines.append((a, -1, b))
    lns = np.array(lines, dtype=np.int64)
    return pts, lns
