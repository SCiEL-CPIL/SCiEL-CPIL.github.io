# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from collections import defaultdict


def _normalize_line(A, B, C):
    """Normalize (A,B,C) so gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(int(A)), gcd(abs(int(B)), abs(int(C))))
    if g > 1:
        A //= g
        B //= g
        C //= g
    # Make first nonzero positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (int(A), int(B), int(C))


def _line_key(A, B, C):
    return _normalize_line(A, B, C)


def _incidence_count(points, lines):
    """Count incidences exactly in int64."""
    n = len(points)
    m = len(lines)
    count = 0
    for p in points:
        x, y = int(p[0]), int(p[1])
        for l in lines:
            A, B, C = int(l[0]), int(l[1]), int(l[2])
            if A * x + B * y + C == 0:
                count += 1
    return count


def _validate(points, lines):
    """Check validity constraints. Return True if valid."""
    # Shape checks
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check entries are integers (within 1e-6)
    for arr, name in [(points, 'points'), (lines, 'lines')]:
        if not np.all(np.isfinite(arr.astype(float))):
            return False
        for row in arr:
            for v in row:
                if not np.isfinite(v) or abs(v - round(v)) > 1e-6:
                    return False
    # Convert to int64
    points = points.astype(np.int64)
    lines = lines.astype(np.int64)
    # No repeated points
    seen = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key in seen:
            return False
        seen.add(key)
    # Line checks
    seen_lines = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        nk = _line_key(A, B, C)
        if nk in seen_lines:
            return False
        seen_lines.add(nk)
    return True


def _generate_candidates_for_points(points, max_lines=27, max_search_time=580.0):
    """Generate top lines by point count for given points, with deduplication."""
    import time
    start_time = time.time()
    
    # Collect all lines through pairs of points
    line_points = defaultdict(list)
    n = len(points)
    for i in range(n):
        xi, yi = int(points[i][0]), int(points[i][1])
        for j in range(i + 1, n):
            xj, yj = int(points[j][0]), int(points[j][1])
            dx = xj - xi
            dy = yj - yi
            # Line through (xi,yi) and (xj,yj): dy*(x-xi) - dx*(y-yi) = 0
            # A=dy, B=-dx, C=dx*yi - dy*xi
            A, B, C = dy, -dx, dx * yi - dy * xi
            key = _line_key(A, B, C)
            line_points[key].append((i, j))
    
    # Also consider single-point "lines"? No, a line is defined by 2 points.
    # But we need to count how many points lie on each line.
    line_to_points = defaultdict(set)
    for key, pairs in line_points.items():
        A, B, C = key
        # For each pair, the line is the same. We need to find all points on this line.
        # To avoid O(n^3), we can just check all points against each unique line.
        # But there could be many unique lines. Let's limit.
        pass
    
    # Better approach: for each unique line, count points on it.
    unique_lines = list(line_points.keys())
    line_counts = {}
    for key in unique_lines:
        A, B, C = key
        cnt = 0
        for p in points:
            x, y = int(p[0]), int(p[1])
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt >= 2:  # At least 2 points define the line
            line_counts[key] = cnt
    
    # Sort by count descending
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Select top lines, ensuring distinctness (already distinct by key)
    selected = []
    for key, cnt in sorted_lines:
        if len(selected) >= max_lines:
            break
        if time.time() - start_time > max_search_time:
            break
        selected.append((key, cnt))
    
    return selected


def _build_hybrid_construction():
    """
    Build the clique-star hybrid construction.
    
    Design:
    - 6 points on vertical line x=0, y in {0,...,5}
    - 48 points on 6x8 grid: x in {1,...,6}, y in {0,...,7}
    - 1 line: x=0 (the star line)
    - 6 lines: x=1,...,6 (verticals of grid)
    - 20 lines: best non-vertical lines through grid points
    
    Total: 6 + 48 = 54 points, 1 + 6 + 20 = 27 lines.
    """
    # Points
    star_points = [(0, y) for y in range(6)]
    grid_points = [(x, y) for x in range(1, 7) for y in range(8)]
    points = np.array(star_points + grid_points, dtype=np.int64)
    
    # Lines
    lines_list = []
    
    # Star line: x = 0 -> (1, 0, 0)
    lines_list.append((1, 0, 0))
    
    # Grid verticals: x = 1, ..., 6 -> (1, 0, -x)
    for x in range(1, 7):
        lines_list.append((1, 0, -x))
    
    # Now find best 20 non-vertical lines through grid points
    # We'll generate all lines through pairs of grid points and count incidences
    grid_only = np.array(grid_points, dtype=np.int64)
    
    line_points_dict = defaultdict(set)
    n = len(grid_only)
    for i in range(n):
        xi, yi = int(grid_only[i][0]), int(grid_only[i][1])
        for j in range(i + 1, n):
            xj, yj = int(grid_only[j][0]), int(grid_only[j][1])
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                continue  # Skip vertical lines (already handled)
            A, B, C = dy, -dx, dx * yi - dy * xi
            key = _line_key(A, B, C)
            line_points_dict[key].add((i, j))
    
    # Count points on each line
    line_counts = {}
    for key, pairs in line_points_dict.items():
        A, B, C = key
        cnt = 0
        for p in grid_only:
            x, y = int(p[0]), int(p[1])
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt >= 2:
            line_counts[key] = cnt
    
    # Sort and pick top 20
    sorted_lines = sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
    for key, cnt in sorted_lines[:20]:
        lines_list.append(key)
    
    lines = np.array(lines_list, dtype=np.int64)
    
    return points, lines


def _local_search(points, lines, max_time=580.0, max_iterations=5000):
    """
    Local search to improve incidences.
    Try moving individual points and swapping lines.
    """
    import time
    start_time = time.time()
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _incidence_count(best_points, best_lines)
    
    # Coordinate descent: try moving each point to nearby lattice positions
    # and try replacing each line with a better one.
    
    # First, optimize lines given fixed points
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
            break
        
        # Try to improve by replacing one line at a time
        improved = False
        for li in range(len(best_lines)):
            # Find best alternative line
            # Generate candidate lines through pairs of points
            # This is expensive, so limit
            pass
        
        if not improved:
            # Try moving points
            for pi in range(len(best_points)):
                if time.time() - start_time > max_time:
                    break
                x0, y0 = int(best_points[pi][0]), int(best_points[pi][1])
                # Try small moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x0 + dx, y0 + dy
                        # Check if this position is already used
                        used = False
                        for pj in range(len(best_points)):
                            if pj != pi:
                                if int(best_points[pj][0]) == new_x and int(best_points[pj][1]) == new_y:
                                    used = True
                                    break
                        if used:
                            continue
                        # Try the move
                        old_val = best_points[pi].copy()
                        best_points[pi] = [new_x, new_y]
                        new_score = _incidence_count(best_points, best_lines)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                        else:
                            best_points[pi] = old_val
    
        if not improved:
            break
    
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    import time
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    # Build the hybrid construction
    points, lines = _build_hybrid_construction()
    
    # Validate
    if not _validate(points, lines):
        # Fallback to simple grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines
    
    score = _incidence_count(points, lines)
    
    # Try local search if time permits
    if time.time() < deadline - 10.0:
        pts2, lines2, score2 = _local_search(points, lines, max_time=deadline - time.time() - 5.0)
        if score2 > score:
            points, lines, score = pts2, lines2, score2
            if not _validate(points, lines):
                # Revert
                points, lines = _build_hybrid_construction()
                score = _incidence_count(points, lines)
    
    return points, lines


# EVOLVE-BLOCK-END
