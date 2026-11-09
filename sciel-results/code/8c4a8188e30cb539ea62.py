# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A, B, C) != (0, 0, 0) else 1
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence_for_line(A, B, C, points):
    count = 0
    for pt in points:
        if A * pt[0] + B * pt[1] + C == 0:
            count += 1
    return count

def _get_all_lines_and_counts(points):
    """Get all distinct lines through pairs of points with their incidence counts."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A, B, C = dy, -dx, dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key not in line_counts:
                line_counts[key] = _count_incidence_for_line(key[0], key[1], key[2], points)
    return line_counts

def _top_k_sum(line_counts, k, exclude=None):
    """Get top-k lines by incidence count, optionally excluding some."""
    if exclude is None:
        exclude = set()
    items = [(k, v) for k, v in line_counts.items() if k not in exclude]
    items.sort(key=lambda x: -x[1])
    return items[:k]

def _evaluate_removal(points, lines, line_idx):
    """Evaluate how many incidences are lost if we remove line line_idx."""
    A, B, C = lines[line_idx]
    lost = 0
    for pt in points:
        if A * pt[0] + B * pt[1] + C == 0:
            # Check if this point is on any other line
            on_other = False
            for li, (A2, B2, C2) in enumerate(lines):
                if li == line_idx:
                    continue
                if A2 * pt[0] + B2 * pt[1] + C2 == 0:
                    on_other = True
                    break
            if not on_other:
                lost += 1
    return lost

def _generate_grid(cols, rows):
    """Generate a cols x rows grid of points."""
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _try_grid(cols, rows, deadline):
    """Try a cols x rows grid configuration."""
    points = _generate_grid(cols, rows)
    n_pts = len(points)
    if n_pts != 54:
        return None
    
    line_counts = _get_all_lines_and_counts(points)
    
    # Grid lines
    grid_lines = []
    for x in range(1, cols + 1):
        grid_lines.append((1, 0, -x))
    for y in range(1, rows + 1):
        grid_lines.append((0, 1, -y))
    
    n_grid = len(grid_lines)
    n_extra = 27 - n_grid
    
    # Get top extra lines
    grid_set = set(grid_lines)
    extra_candidates = _top_k_sum(line_counts, n_extra + 50, exclude=grid_set)
    
    # Start with top n_extra
    selected_extra = [line for line, _ in extra_candidates[:n_extra]]
    
    # Local search: try swapping each selected line with a non-selected one
    improved = True
    while improved:
        improved = False
        all_extra = [line for line, _ in extra_candidates]
        for i in range(len(selected_extra)):
            for j, candidate in enumerate(all_extra):
                if candidate in selected_extra:
                    continue
                # Try swap
                new_selected = selected_extra[:]
                new_selected[i] = candidate
                # Count incidences
                total = 0
                for line in grid_lines + new_selected:
                    total += line_counts.get(line, 0)
                old_total = sum(line_counts.get(line, 0) for line in grid_lines + selected_extra)
                if total > old_total:
                    selected_extra = new_selected
                    improved = True
                    break
            if improved:
                break
    
    all_lines = grid_lines + selected_extra
    lines = np.array(all_lines, dtype=np.int64)
    
    # Verify
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    unique_pts = set(map(tuple, points))
    assert len(unique_pts) == 54
    norm_lines = set()
    for l in lines:
        nl = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        assert nl not in norm_lines
        norm_lines.add(nl)
    
    # Count incidences
    total_inc = 0
    for pt in points:
        for l in lines:
            if l[0] * pt[0] + l[1] * pt[1] + l[2] == 0:
                total_inc += 1
    
    return points, lines, total_inc

def _try_rect_with_extras(cols, rows, deadline):
    """Try a rectangular grid with optimized extras using more thorough search."""
    points = _generate_grid(cols, rows)
    n_pts = len(points)
    if n_pts != 54:
        return None
    
    line_counts = _get_all_lines_and_counts(points)
    
    # Grid lines
    grid_lines = []
    for x in range(1, cols + 1):
        grid_lines.append((1, 0, -x))
    for y in range(1, rows + 1):
        grid_lines.append((0, 1, -y))
    
    n_grid = len(grid_lines)
    n_extra = 27 - n_grid
    
    # Get all non-grid lines sorted by count
    grid_set = set(grid_lines)
    all_extra = [(line, cnt) for line, cnt in line_counts.items() if line not in grid_set]
    all_extra.sort(key=lambda x: -x[1])
    
    # Greedy selection with consideration of overlap
    # Use a smarter approach: select lines that maximize marginal gain
    selected_extra = []
    remaining = list(all_extra)
    
    for _ in range(n_extra):
        best_line = None
        best_gain = -1
        for line, cnt in remaining:
            # Marginal gain = number of points on this line not already covered
            A, B, C = line
            covered = 0
            for pt in points:
                if A * pt[0] + B * pt[1] + C == 0:
                    # Check if already covered by grid or selected extra
                    is_covered = False
                    for gl in grid_lines:
                        if gl[0] * pt[0] + gl[1] * pt[1] + gl[2] == 0:
                            is_covered = True
                            break
                    if not is_covered:
                        for sl in selected_extra:
                            if sl[0] * pt[0] + sl[1] * pt[1] + sl[2] == 0:
                                is_covered = True
                                break
                    if not is_covered:
                        covered += 1
            if covered > best_gain:
                best_gain = covered
                best_line = line
        if best_line is not None:
            selected_extra.append(best_line)
            remaining = [r for r in remaining if r[0] != best_line]
    
    all_lines = grid_lines + selected_extra
    lines = np.array(all_lines, dtype=np.int64)
    
    # Verify
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    unique_pts = set(map(tuple, points))
    assert len(unique_pts) == 54
    norm_lines = set()
    for l in lines:
        nl = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        assert nl not in norm_lines
        norm_lines.add(nl)
    
    # Count incidences
    total_inc = 0
    for pt in points:
        for l in lines:
            if l[0] * pt[0] + l[1] * pt[1] + l[2] == 0:
                total_inc += 1
    
    return points, lines, total_inc

def _try_lattice_points(deadline):
    """Try non-rectangular lattice point configurations."""
    # Try 54 points from a larger grid with some removed
    # This is more complex; skip for now if time is tight
    return None

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    best_result = None
    best_score = 0
    
    # Try different grid configurations
    configs = [
        (6, 9),   # 6 cols x 9 rows
        (9, 6),   # 9 cols x 6 rows
        (3, 18),  # 3 cols x 18 rows
        (18, 3),  # 18 cols x 3 rows
        (2, 27),  # 2 cols x 27 rows
        (27, 2),  # 27 cols x 2 rows
    ]
    
    for cols, rows in configs:
        if time.time() > deadline - 60:
            break
        
        # Method 1: Top-k with local search
        result = _try_grid(cols, rows, deadline)
        if result is not None:
            pts, lns, inc = result
            if inc > best_score:
                best_score = inc
                best_result = (pts, lns)
        
        # Method 2: Greedy with marginal gain
        if time.time() > deadline - 30:
            break
        result2 = _try_rect_with_extras(cols, rows, deadline)
        if result2 is not None:
            pts, lns, inc = result2
            if inc > best_score:
                best_score = inc
                best_result = (pts, lns)
    
    if best_result is None:
        # Fallback: simple 6x9 grid
        points = _generate_grid(6, 9)
        grid_lines = []
        for x in range(1, 7):
            grid_lines.append((1, 0, -x))
        for y in range(1, 10):
            grid_lines.append((0, 1, -y))
        line_counts = _get_all_lines_and_counts(points)
        grid_set = set(grid_lines)
        extra = _top_k_sum(line_counts, 12, exclude=grid_set)
        all_lines = grid_lines + [l for l, _ in extra]
        lines = np.array(all_lines, dtype=np.int64)
        best_result = (points, lines)
    
    return best_result

# EVOLVE-BLOCK-END
