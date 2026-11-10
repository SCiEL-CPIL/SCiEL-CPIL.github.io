# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_points(p1, p2):
    """Get normalized line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A == 0 and B == 0:
        return None
    return _normalize_line(A, B, C)


def _count_incidence(points, line):
    """Count how many points lie on the line A*x + B*y + C = 0."""
    A, B, C = line
    # Vectorized check
    vals = A * points[:, 0] + B * points[:, 1] + C
    return np.sum(vals == 0)


def _greedy_select_lines(points, n_lines, candidate_lines):
    """Greedy select n_lines from candidate_lines to maximize total incidences."""
    n_points = len(points)
    # Precompute incidence for each candidate line
    line_inc = np.zeros(len(candidate_lines), dtype=np.int64)
    for i, line in enumerate(candidate_lines):
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        line_inc[i] = np.sum(vals == 0)
    
    selected = []
    selected_set = set()
    total_inc = 0
    
    for _ in range(n_lines):
        best_idx = -1
        best_inc = -1
        for i in range(len(candidate_lines)):
            if i in selected_set:
                continue
            if line_inc[i] > best_inc:
                best_inc = line_inc[i]
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        total_inc += best_inc
    
    return [candidate_lines[i] for i in selected], total_inc


def _generate_candidate_lines(points):
    """Generate candidate lines from a set of points."""
    n = len(points)
    lines_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line is not None:
                lines_set.add(line)
    return list(lines_set)


def _evaluate_grid(rows, cols, x_start=1, y_start=1):
    """Evaluate a grid configuration."""
    points_list = []
    for x in range(x_start, x_start + cols):
        for y in range(y_start, y_start + rows):
            points_list.append((x, y))
    points = np.array(points_list, dtype=np.int64)
    
    # Generate candidate lines
    candidate_lines = _generate_candidate_lines(points)
    
    # Greedy select 27 lines
    selected_lines, total_inc = _greedy_select_lines(points, 27, candidate_lines)
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines, total_inc


def _try_factorization(n, m):
    """Try all factorizations of n and find the best grid."""
    best = None
    best_inc = -1
    best_points = None
    best_lines = None
    
    for cols in range(1, n + 1):
        if n % cols != 0:
            continue
        rows = n // cols
        # Try different offsets
        for x_start in [0, 1]:
            for y_start in [0, 1]:
                points, lines, inc = _evaluate_grid(rows, cols, x_start, y_start)
                if inc > best_inc:
                    best_inc = inc
                    best_points = points
                    best_lines = lines
    
    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    Uses grid-based construction with greedy line selection.
    """
    start_time = time.time()
    deadline = 630 - 30  # Safety margin
    
    n_points = 54
    n_lines = 27
    
    # Try all factorizations of 54
    best_points = None
    best_lines = None
    best_inc = -1
    
    for cols in range(1, n_points + 1):
        if n_points % cols != 0:
            continue
        rows = n_points // cols
        
        # Skip if too many columns (would be very thin, fewer diagonals)
        # Actually, let's try all
        
        for x_start in [0, 1]:
            for y_start in [0, 1]:
                if time.time() > deadline:
                    break
                
                points, lines, inc = _evaluate_grid(rows, cols, x_start, y_start)
                if inc > best_inc:
                    best_inc = inc
                    best_points = points
                    best_lines = lines
        
        if time.time() > deadline:
            break
    
    if best_points is None:
        # Fallback to the original Szemeredi-Trotter construction
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows_list = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows_list.append((a, -1, b))
        best_lines = np.array(rows_list, dtype=np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
