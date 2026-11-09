# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _compute_incidences(points, lines):
    """Compute total incidences."""
    total = 0
    for (A, B, C) in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.count_nonzero(vals == 0))
    return total


def _best_lines_for_points(points, n_lines=27):
    """Find the n_lines best lines for a given point set."""
    n = len(points)
    line_set = set()
    
    for i in range(n):
        xi, yi = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(points[j, 0]), int(points[j, 1])
            A = yi - yj
            B = xj - xi
            C = -(xi * yj - xj * yi)
            key = _normalize(A, B, C)
            line_set.add(key)
    
    # Count incidences for each line
    line_counts = {}
    for key in line_set:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        line_counts[key] = int(np.count_nonzero(vals == 0))
    
    # Sort by count descending, then by key for determinism
    candidates = sorted(line_counts.items(), key=lambda t: (-t[1], t[0]))
    
    selected = []
    for key, cnt in candidates:
        if cnt < 2:
            break
        selected.append(key)
        if len(selected) == n_lines:
            break
    
    if len(selected) < n_lines:
        # Top up with any remaining
        for key, cnt in candidates:
            if key not in selected:
                selected.append(key)
                if len(selected) == n_lines:
                    break
    
    return selected


def _try_grid(rows, cols):
    """Try a rectangular grid of given dimensions."""
    xs = np.arange(1, cols + 1, dtype=np.int64)
    ys = np.arange(1, rows + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) != 54:
        return None, None, 0
    
    lines_list = _best_lines_for_points(points, 27)
    if len(lines_list) < 27:
        return None, None, 0
    
    lines = np.array(lines_list, dtype=np.int64)
    incidences = _compute_incidences(points, lines)
    return points, lines, incidences


def _try_kxk_plus_extras(k, extras):
    """Try a k×k grid plus extra points placed to maximize incidences."""
    base_pts = []
    for x in range(1, k + 1):
        for y in range(1, k + 1):
            base_pts.append((x, y))
    
    points = list(base_pts)
    
    # Try adding extras at various positions to create new high-incidence lines
    # We'll try a few candidate placements
    best_result = None
    best_inc = -1
    
    # Candidate extra positions: extend the grid in various ways
    candidate_positions = []
    for x in range(0, k + 3):
        for y in range(0, k + 3):
            if (x, y) not in base_pts and 1 <= x <= k + 2 and 1 <= y <= k + 2:
                candidate_positions.append((x, y))
    
    # Also try extending one axis
    for x in range(1, k + 1):
        candidate_positions.append((x, k + 1))
        candidate_positions.append((x, 0))
    for y in range(1, k + 1):
        candidate_positions.append((k + 1, y))
        candidate_positions.append((0, y))
    
    # Remove duplicates
    seen = set()
    unique_candidates = []
    for p in candidate_positions:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)
    
    # Try different subsets of extras
    # For simplicity, try greedy: add one extra at a time that maximizes total incidences
    for attempt in range(min(len(extras), len(unique_candidates))):
        best_pos = None
        best_total = -1
        for pos in unique_candidates:
            if pos in points:
                continue
            test_points = points + [pos]
            test_points_arr = np.array(test_points, dtype=np.int64)
            test_lines = _best_lines_for_points(test_points_arr, 27)
            if len(test_lines) < 27:
                continue
            test_lines_arr = np.array(test_lines, dtype=np.int64)
            inc = _compute_incidences(test_points_arr, test_lines_arr)
            if inc > best_total:
                best_total = inc
                best_pos = pos
        if best_pos is None:
            break
        points.append(best_pos)
    
    if len(points) != 54:
        return None, None, 0
    
    points_arr = np.array(points, dtype=np.int64)
    lines_list = _best_lines_for_points(points_arr, 27)
    if len(lines_list) < 27:
        return None, None, 0
    
    lines = np.array(lines_list, dtype=np.int64)
    incidences = _compute_incidences(points_arr, lines)
    
    if best_result is None or incidences > best_inc:
        best_result = (points_arr, lines, incidences)
        best_inc = incidences
    
    return best_result[0], best_result[1], best_result[2] if best_result else (None, None, 0)


def szemeredi_trotter_incidences():
    t_start = time.time()
    deadline = 600.0
    margin = 30.0
    
    best_points = None
    best_lines = None
    best_inc = -1
    
    # Try various rectangular grids that multiply to 54
    grid_dims = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    
    for rows, cols in grid_dims:
        if time.time() > deadline - margin:
            break
        pts, lns, inc = _try_grid(rows, cols)
        if pts is not None and inc > best_inc:
            best_inc = inc
            best_points = pts
            best_lines = lns
    
    # Try k×k grids plus extras
    for k in [6, 7]:
        if time.time() > deadline - margin:
            break
        extras = 54 - k * k
        if extras < 0 or extras > 10:  # Keep extras manageable
            continue
        pts, lns, inc = _try_kxk_plus_extras(k, extras)
        if pts is not None and inc > best_inc:
            best_inc = inc
            best_points = pts
            best_lines = lns
    
    # Also try a "fat" grid: 7×8 = 56, remove 2 points strategically
    if time.time() <= deadline - margin:
        # 7×8 grid
        xs = np.arange(1, 9, dtype=np.int64)
        ys = np.arange(1, 8, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        full_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Remove 2 points that are on the fewest high-incidence lines
        # Try removing each pair and see which gives best total
        # For speed, try removing corner points first
        corner_indices = [0, 7, 48, 55]  # corners of 7x8
        # Also try removing points that break fewer lines
        # Simplified: try a few removal strategies
        
        # Strategy 1: remove two random-ish points (deterministic)
        for rem1 in range(0, 56, 7):  # remove from different rows
            for rem2 in range(rem1 + 1, 56, 7):
                if time.time() > deadline - margin:
                    break
                mask = np.ones(56, dtype=bool)
                mask[rem1] = False
                mask[rem2] = False
                test_pts = full_pts[mask]
                if len(test_pts) != 54:
                    continue
                test_lines = _best_lines_for_points(test_pts, 27)
                if len(test_lines) < 27:
                    continue
                test_lines_arr = np.array(test_lines, dtype=np.int64)
                inc = _compute_incidences(test_pts, test_lines_arr)
                if inc > best_inc:
                    best_inc = inc
                    best_points = test_pts
                    best_lines = test_lines_arr
            if time.time() > deadline - margin:
                break
    
    # Fallback
    if best_points is None or best_lines is None:
        # Use the 6×9 grid as guaranteed fallback
        pts, lns, _ = _try_grid(9, 6)
        best_points = pts
        best_lines = lns
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
