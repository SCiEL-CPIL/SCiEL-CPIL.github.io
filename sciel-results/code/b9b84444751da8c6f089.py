import numpy as np
import time
from math import gcd
from itertools import combinations
from collections import defaultdict

def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so first nonzero is positive, divide by gcd."""
    g = gcd(gcd(abs(int(A)), abs(int(B))), abs(int(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (int(A), int(B), int(C))

def _line_incidence_count(points, line):
    """Count how many points lie on the line A*x + B*y + C = 0."""
    A, B, C = line
    return int(np.sum(A * points[:, 0] + B * points[:, 1] + C == 0))

def _find_best_lines(points, num_lines, time_limit=None, start_time=None):
    """Find the best num_lines lines maximizing total incidences."""
    if start_time is None:
        start_time = time.time()
    
    n = len(points)
    px = points[:, 0]
    py = points[:, 1]
    
    # Build all unique lines through pairs of points
    line_counts = defaultdict(int)  # normalized line -> count
    line_points = defaultdict(set)  # normalized line -> set of point indices
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = px[i], py[i]
            x2, y2 = px[j], py[j]
            # Line through (x1,y1) and (x2,y2): (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            line_counts[key] += 1
            line_points[key].add(i)
            line_points[key].add(j)
    
    # For lines with count > 1, verify by checking all points
    # For lines with count == 1 (only one pair), they have exactly 2 points
    # We need to get the actual count for each line
    
    # Actually, line_counts counts pairs, not points. A line with k points has C(k,2) pairs.
    # So we need to recover k from the pair count.
    # C(k,2) = k*(k-1)/2
    # For small k: 1 pair -> 2 pts, 3 pairs -> 3 pts, 6 pairs -> 4 pts, 10 pairs -> 5 pts, etc.
    
    # Better approach: for each unique line, count actual points on it
    candidate_lines = list(line_counts.keys())
    
    # Sort by pair count descending (proxy for point count)
    candidate_lines.sort(key=lambda l: line_counts[l], reverse=True)
    
    # For efficiency, only consider lines that pass through at least 2 points
    # and compute exact point counts
    exact_counts = {}
    for line in candidate_lines:
        A, B, C = line
        count = int(np.sum(A * px + B * py + C == 0))
        exact_counts[line] = count
    
    # Sort by exact count descending
    sorted_lines = sorted(exact_counts.keys(), key=lambda l: exact_counts[l], reverse=True)
    
    # Greedy selection: pick lines that maximize marginal gain
    # But greedy might not be optimal. Let's try a few strategies.
    
    # Strategy 1: Just take top num_lines by count
    top_lines = sorted_lines[:num_lines]
    total_inc = sum(exact_counts[l] for l in top_lines)
    
    # Strategy 2: Greedy with marginal gain
    selected = []
    selected_set = set()
    remaining_inc = {l: exact_counts[l] for l in sorted_lines}
    
    for _ in range(num_lines):
        best_line = None
        best_gain = -1
        for l in sorted_lines:
            if l in selected_set:
                continue
            # Marginal gain: how many new points does this line cover?
            # We need to track which points are already covered
            # This is expensive. Let's just use the count as a proxy.
            gain = remaining_inc[l]
            if gain > best_gain:
                best_gain = gain
                best_line = l
        if best_line is not None:
            selected.append(best_line)
            selected_set.add(best_line)
    
    total_inc_greedy = sum(exact_counts[l] for l in selected)
    
    # Pick the better strategy
    if total_inc_greedy >= total_inc:
        return selected, total_inc_greedy
    else:
        return top_lines, total_inc

def _evaluate_configuration(points, num_lines=27):
    """Evaluate a point configuration by finding best lines."""
    if len(points) != 54:
        return -1, None
    lines, inc = _find_best_lines(points, num_lines)
    if lines is None:
        return -1, None
    return inc, lines

def _make_grid_points(nx, ny, remove=()):
    """Create an nx x ny grid with some points removed. Returns (54, 2) array or None."""
    xs = np.arange(1, nx + 1)
    ys = np.arange(1, ny + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if remove:
        mask = np.ones(len(points), dtype=bool)
        for (rx, ry) in remove:
            idx = np.where((points[:, 0] == rx) & (points[:, 1] == ry))[0]
            if len(idx) > 0:
                mask[idx[0]] = False
        points = points[mask]
    
    if len(points) != 54:
        return None
    return points

def _make_cross_points(center_x, center_y, arms, per_arm):
    """Create a cross/plus shape."""
    points = []
    for dx in range(-arms, arms + 1):
        for dy in range(-arms, arms + 1):
            if abs(dx) + abs(dy) <= per_arm:
                points.append((center_x + dx, center_y + dy))
    return np.array(points, dtype=np.int64)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    best_inc = -1
    best_points = None
    best_lines = None
    
    # Predefined configurations to try
    configs = []
    
    # 6x9 grid (current best)
    configs.append(('6x9', _make_grid_points(6, 9)))
    
    # 7x8 with 2 removed
    for r1 in [(7, 8), (7, 7), (7, 1), (7, 2), (1, 8), (1, 7), (1, 1), (1, 2)]:
        for r2 in [(7, 8), (7, 7), (7, 6), (7, 5), (7, 1), (7, 2), (1, 8), (1, 7), (1, 1), (1, 2)]:
            if r1 != r2:
                configs.append((f'7x8-{r1}-{r2}', _make_grid_points(7, 8, remove=[r1, r2])))
    
    # 5x11 with 1 removed
    for r in [(5, 11), (5, 10), (5, 1), (5, 2), (1, 11), (1, 10), (1, 1), (1, 2)]:
        configs.append((f'5x11-{r}', _make_grid_points(5, 11, remove=[r])))
    
    # 9x6 grid
    configs.append(('9x6', _make_grid_points(9, 6)))
    
    # 8x7 with 2 removed
    for r1 in [(8, 7), (8, 6), (8, 1), (1, 7), (1, 6), (1, 1)]:
        for r2 in [(8, 7), (8, 6), (8, 5), (8, 1), (1, 7), (1, 6), (1, 1)]:
            if r1 != r2:
                configs.append((f'8x7-{r1}-{r2}', _make_grid_points(8, 7, remove=[r1, r2])))
    
    # 4x14 with 2 removed
    for r1 in [(4, 14), (4, 13), (4, 1), (1, 14), (1, 13), (1, 1)]:
        for r2 in [(4, 14), (4, 13), (4, 12), (4, 1), (1, 14), (1, 13), (1, 1)]:
            if r1 != r2:
                configs.append((f'4x14-{r1}-{r2}', _make_grid_points(4, 14, remove=[r1, r2])))
    
    # 10x6 with 6 removed
    for r in [(10, 6), (10, 5), (10, 4), (10, 3), (10, 2), (10, 1)]:
        configs.append((f'10x6-{r}', _make_grid_points(10, 6, remove=[r])))
    
    # 12x5 with 6 removed
    for r in [(12, 5), (12, 4), (12, 3), (12, 2), (12, 1)]:
        configs.append((f'12x5-{r}', _make_grid_points(12, 5, remove=[r])))
    
    # 3x18 with 6 removed
    for r in [(3, 18), (3, 17), (3, 1), (2, 18), (2, 17), (1, 18)]:
        configs.append((f'3x18-{r}', _make_grid_points(3, 18, remove=[r])))
    
    # 18x3 with 6 removed
    for r in [(18, 3), (17, 3), (1, 3), (18, 2), (17, 2), (18, 1)]:
        configs.append((f'18x3-{r}', _make_grid_points(18, 3, remove=[r])))
    
    # Evaluate each configuration
    for name, points in configs:
        if points is None:
            continue
        if time.time() > deadline - 30:
            break
        
        inc, lines = _evaluate_configuration(points, 27)
        if inc > best_inc:
            best_inc = inc
            best_points = points.copy()
            best_lines = lines
    
    # If no configuration worked, fall back to 6x9 grid
    if best_points is None:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        lines_list = []
        for i in range(1, 7):
            lines_list.append((1, 0, -i))
        for j in range(1, 10):
            lines_list.append((0, 1, -j))
        lines_list.append((1, -1, 0))
        lines_list.append((1, -1, 1))
        lines_list.append((1, -1, 2))
        lines_list.append((1, -1, 3))
        lines_list.append((1, -1, -1))
        lines_list.append((1, -1, 4))
        lines_list.append((1, -1, -2))
        lines_list.append((2, -1, -1))
        lines_list.append((2, -1, 0))
        lines_list.append((2, -1, 1))
        lines_list.append((2, -1, -2))
        lines_list.append((2, -1, 2))
        best_lines = lines_list
    
    # Ensure best_lines is a list of tuples
    if isinstance(best_lines, np.ndarray):
        best_lines = [tuple(row) for row in best_lines]
    
    best_lines = np.array(best_lines, dtype=np.int64)
    
    return best_points, best_lines
