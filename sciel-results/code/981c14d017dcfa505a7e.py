# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _count_incidences(points, lines):
    """Count exact integer incidences."""
    # points: (n, 2) int64, lines: (m, 3) int64
    # For each line (A,B,C) and point (x,y): A*x + B*y + C == 0
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(m):
        A, B, C = lines[i]
        # Vectorized check for this line
        vals = A * points[:, 0] + B * points[:, 1] + C
        count += np.sum(vals == 0)
    return count

def _normalize_line(A, B, C):
    """Normalize a line so first nonzero entry is positive, divided by gcd."""
    # Find gcd of |A|, |B|, |C|
    g = 0
    for v in [abs(int(A)), abs(int(B)), abs(int(C))]:
        g = gcd(g, v)
    if g == 0:
        return (0, 0, 0)
    A, B, C = int(A) // g, int(B) // g, int(C) // g
    # Make first nonzero positive
    if A != 0:
        if A < 0:
            A, B, C = -A, -B, -C
    elif B != 0:
        if B < 0:
            A, B, C = -A, -B, -C
    else:
        if C < 0:
            A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_two_points(p1, p2):
    """Get normalized line through two points."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _find_best_lines_for_points(points, num_lines=27):
    """Find the best num_lines lines that maximize incidences with given points."""
    n = len(points)
    # Collect all lines determined by pairs of points
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_two_points(points[i], points[j])
            if line[0] == 0 and line[1] == 0:
                continue
            line_counts[line] = line_counts.get(line, 0) + 1
    
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Greedily pick lines, but avoid duplicates
    selected = []
    selected_set = set()
    for line, count in sorted_lines:
        if len(selected) >= num_lines:
            break
        if line not in selected_set:
            selected.append(line)
            selected_set.add(line)
    
    # If we don't have enough, add lines with 0 points (shouldn't happen normally)
    while len(selected) < num_lines:
        # Add a line that doesn't hit any points
        selected.append((1, 1, -(10**6 + len(selected))))
    
    return np.array(selected, dtype=np.int64)

def _local_search(points, lines, max_iters=5000, deadline=None):
    """Local search to improve incidences by moving points and adjusting lines."""
    n = len(points)
    m = len(lines)
    best_inc = _count_incidences(points, lines)
    current_points = points.copy()
    current_lines = lines.copy()
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try moving a random point
        idx = rng.integers(0, n)
        old_x, old_y = int(current_points[idx, 0]), int(current_points[idx, 1])
        
        # Try small moves
        dx = rng.integers(-3, 4)
        dy = rng.integers(-3, 4)
        new_x, new_y = old_x + dx, old_y + dy
        
        # Check if point already exists
        new_point = np.array([new_x, new_y], dtype=np.int64)
        if np.any(np.all(current_points == new_point, axis=1)):
            continue
        
        # Temporarily move point
        current_points[idx, 0] = new_x
        current_points[idx, 1] = new_y
        
        # Recompute best lines for this point set
        new_lines = _find_best_lines_for_points(current_points, m)
        new_inc = _count_incidences(current_points, new_lines)
        
        if new_inc >= best_inc:
            best_inc = new_inc
            current_lines = new_lines
        else:
            # Revert
            current_points[idx, 0] = old_x
            current_points[idx, 1] = old_y
    
    return current_points, current_lines, best_inc

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    
    Strategy:
    1. Start with a 6x9 grid of points and optimized lines (176 incidences).
    2. Run local search to improve by moving points and re-optimizing lines.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # 1. Generate initial Points (6x9 grid)
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate initial Lines (optimized for grid)
    lines = []
    for c in range(1, 10):
        lines.append((0, 1, -c))
    for c in range(1, 7):
        lines.append((1, 0, -c))
    best_c_slope_1 = [0, 1, 2, 3, -1, 4]
    for c in best_c_slope_1:
        lines.append((-1, 1, -c))
    best_c_slope_neg_1 = [7, 8, 9, 10, 6, 11]
    for c in best_c_slope_neg_1:
        lines.append((1, 1, -c))
    lines = np.array(lines, dtype=np.int64)
    
    initial_inc = _count_incidences(points, lines)
    
    # 3. Local search
    points, lines, best_inc = _local_search(points, lines, max_iters=10000, deadline=deadline)
    
    # Ensure validity: no duplicate points, no invalid lines
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != len(points):
        # Fall back to grid
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array(lines_list, dtype=np.int64) if 'lines_list' in dir() else np.array([
            (0, 1, -c) for c in range(1, 10)
        ] + [(1, 0, -c) for c in range(1, 7)] +
        [(-1, 1, -c) for c in [0, 1, 2, 3, -1, 4]] +
        [(1, 1, -c) for c in [7, 8, 9, 10, 6, 11]], dtype=np.int64)
    
    # Check for invalid lines (A=B=0)
    valid_mask = ~((lines[:, 0] == 0) & (lines[:, 1] == 0))
    if not np.all(valid_mask):
        # Fall back to grid
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([
            (0, 1, -c) for c in range(1, 10)
        ] + [(1, 0, -c) for c in range(1, 7)] +
        [(-1, 1, -c) for c in [0, 1, 2, 3, -1, 4]] +
        [(1, 1, -c) for c in [7, 8, 9, 10, 6, 11]], dtype=np.int64)
    
    # Check for duplicate lines
    normalized = [_normalize_line(*tuple(line)) for line in lines]
    seen = set()
    unique_normalized = []
    for nl in normalized:
        if nl not in seen:
            seen.add(nl)
            unique_normalized.append(nl)
    if len(unique_normalized) != len(normalized):
        # Fall back to grid
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([
            (0, 1, -c) for c in range(1, 10)
        ] + [(1, 0, -c) for c in range(1, 7)] +
        [(-1, 1, -c) for c in [0, 1, 2, 3, -1, 4]] +
        [(1, 1, -c) for c in [7, 8, 9, 10, 6, 11]], dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
