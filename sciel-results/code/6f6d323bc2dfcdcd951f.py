import numpy as np
import time
import random
from math import gcd
from itertools import combinations


def _count_incidences(points, lines):
    """Count point-line incidences exactly using int64 arithmetic."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _line_key(A, B, C):
    """Canonical form of a line for deduplication."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Compute line (A,B,C) through two integer points, canonicalized."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _line_key(A, B, C)


def _enumerate_best_lines(points, k=27):
    """
    Enumerate all lines through pairs of points, count incidences for each,
    and return the top-k distinct lines.
    """
    n = len(points)
    # Map from canonical line key to incidence count
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            if key is None:
                continue
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Now for each unique line, count total incidences (how many points lie on it)
    # The count above is the number of pairs, but we need actual point incidences
    # A line through points i,j has incidence = number of points on that line
    # We can compute this by checking all points
    
    # Sort by pair count as proxy, then verify
    # Actually, let's just compute incidences directly for candidate lines
    # Get top candidates by pair count first
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Take top 200 candidates and compute exact incidences
    candidates = []
    for key, pair_count in sorted_lines[:300]:
        A, B, C = key
        inc = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                inc += 1
        candidates.append((inc, key))
    
    # Sort by incidence count
    candidates.sort(key=lambda x: -x[0])
    
    # Pick top k distinct lines
    selected = []
    seen = set()
    for inc, key in candidates:
        if key not in seen:
            selected.append(key)
            seen.add(key)
            if len(selected) == k:
                break
    
    # If we don't have enough, fill with random valid lines
    if len(selected) < k:
        # Add lines through single points with random slopes
        random.seed(42)
        attempts = 0
        while len(selected) < k and attempts < 10000:
            attempts += 1
            pt = random.choice(points)
            x, y = pt
            A = random.randint(-3, 3)
            B = random.randint(-3, 3)
            if A == 0 and B == 0:
                A = 1
            C = -(A * x + B * y)
            key = _line_key(A, B, C)
            if key is not None and key not in seen:
                selected.append(key)
                seen.add(key)
    
    return [list(line) for line in selected[:k]]


def _try_grid_config(rows, cols):
    """
    Try a rows x cols grid, enumerate best lines.
    Returns (points, lines, incidence_count).
    """
    points_list = [(x, y) for y in range(rows) for x in range(cols)]
    points = np.array(points_list, dtype=np.int64)
    
    lines = _enumerate_best_lines(points, k=27)
    lines = np.array(lines, dtype=np.int64)
    
    inc = _count_incidences(points, lines)
    return points, lines, inc


def _try_staggered_grid(rows, cols, offset=1):
    """
    Staggered grid: even rows shifted by offset.
    """
    points_list = []
    for y in range(rows):
        shift = offset if y % 2 == 1 else 0
        for x in range(cols):
            points_list.append((x + shift, y))
    points = np.array(points_list, dtype=np.int64)
    
    lines = _enumerate_best_lines(points, k=27)
    lines = np.array(lines, dtype=np.int64)
    
    inc = _count_incidences(points, lines)
    return points, lines, inc


def _try_lattice_config(a, b, c, d):
    """
    Lattice points: (x*a + y*c, x*b + y*d) for x in range, y in range
    """
    # Generate lattice points
    points_list = []
    for x in range(6):
        for y in range(9):
            px = x * a + y * c
            py = x * b + y * d
            points_list.append((px, py))
    points = np.array(points_list, dtype=np.int64)
    
    # Check for duplicates
    if len(set(map(tuple, points.tolist()))) < 54:
        return None, None, 0
    
    lines = _enumerate_best_lines(points, k=27)
    lines = np.array(lines, dtype=np.int64)
    
    inc = _count_incidences(points, lines)
    return points, lines, inc


def _local_search_lines(points, lines, time_limit=60.0, start_time=None):
    """
    Local search: try replacing one line with another high-incidence line.
    """
    if start_time is None:
        start_time = time.time()
    
    current_inc = _count_incidences(points, lines)
    best_inc = current_inc
    best_lines = lines.copy()
    
    # Precompute all candidate lines with their incidences
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            if key is not None:
                line_counts[key] = line_counts.get(key, 0) + 1
    
    # Compute exact incidences for all candidates
    candidate_incs = {}
    for key in line_counts:
        A, B, C = key
        inc = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                inc += 1
        candidate_incs[key] = inc
    
    # Sort candidates by incidence
    sorted_candidates = sorted(candidate_incs.items(), key=lambda x: -x[1])
    
    # Current line keys
    current_keys = [_line_key(lines[i][0], lines[i][1], lines[i][2]) for i in range(27)]
    
    temp = 3.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        line_idx = random.randint(0, 26)
        old_key = current_keys[line_idx]
        
        # Pick a random candidate from top 100
        if sorted_candidates:
            max_idx = min(100, len(sorted_candidates))
            new_key = sorted_candidates[random.randint(0, max_idx - 1)][0]
        else:
            continue
        
        if new_key == old_key:
            continue
        
        # Check if new_key is already used by another line
        if new_key in current_keys and current_keys.index(new_key) != line_idx:
            continue
        
        # Swap
        current_keys[line_idx] = new_key
        lines[line_idx] = list(new_key)
        
        new_inc = _count_incidences(points, lines)
        delta = new_inc - current_inc
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 0.1)):
            current_inc = new_inc
            if current_inc > best_inc:
                best_inc = current_inc
                best_lines = lines.copy()
        else:
            # Revert
            lines[line_idx] = list(old_key)
            current_keys[line_idx] = old_key
        
        temp *= cooling
    
    return best_lines, best_inc


def _try_rectangular_grid_with_search(rows, cols, time_limit=60.0, start_time=None):
    """
    Try a rows x cols grid, enumerate best lines, then local search.
    """
    if start_time is None:
        start_time = time.time()
    
    points_list = [(x, y) for y in range(rows) for x in range(cols)]
    points = np.array(points_list, dtype=np.int64)
    
    lines = _enumerate_best_lines(points, k=27)
    lines = np.array(lines, dtype=np.int64)
    
    inc = _count_incidences(points, lines)
    
    # Local search if time permits
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5:
        best_lines, best_inc = _local_search_lines(points, lines, time_limit=remaining - 2, start_time=time.time())
        if best_inc > inc:
            lines = best_lines
            inc = best_inc
    
    return points, lines, inc


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Strategy: try multiple grid/lattice configurations, enumerate best lines,
    then local search.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    best_points = None
    best_lines = None
    best_inc = 0
    
    random.seed(12345)
    
    # Try various grid configurations
    configs = [
        ("6x9 grid", lambda: _try_rectangular_grid_with_search(9, 6, time_limit=80, start_time=start_time)),
        ("9x6 grid", lambda: _try_rectangular_grid_with_search(6, 9, time_limit=80, start_time=start_time)),
        ("3x18 grid", lambda: _try_rectangular_grid_with_search(18, 3, time_limit=60, start_time=start_time)),
        ("18x3 grid", lambda: _try_rectangular_grid_with_search(3, 18, time_limit=60, start_time=start_time)),
        ("Staggered 9x6", lambda: _try_staggered_grid(9, 6, offset=1)),
        ("Staggered 6x9", lambda: _try_staggered_grid(6, 9, offset=1)),
    ]
    
    for name, func in configs:
        if time.time() > deadline - 30:
            break
        try:
            pts, lns, inc = func()
            if pts is not None and inc > best_inc:
                best_inc = inc
                best_points = pts.copy()
                best_lines = lns.copy()
        except Exception:
            continue
    
    # Try some lattice configurations
    lattice_configs = [
        (1, 1, 0, 1),   # Standard grid
        (1, 1, 1, 1),   # Diagonal shift
        (1, 1, 0, 2),   # Double vertical
        (1, 1, 2, 1),   # Double horizontal
        (1, 2, 0, 1),   # Stretched vertical
        (2, 1, 0, 1),   # Stretched horizontal
        (1, 1, 1, 2),   # Diagonal
        (1, 2, 1, 1),   # Shear
    ]
    
    for a, b, c, d in lattice_configs:
        if time.time() > deadline - 20:
            break
        try:
            pts, lns, inc = _try_lattice_config(a, b, c, d)
            if pts is not None and inc > best_inc:
                best_inc = inc
                best_points = pts.copy()
                best_lines = lns.copy()
        except Exception:
            continue
    
    # Final local search on best configuration if time permits
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        try:
            best_lines_ls, best_inc_ls = _local_search_lines(
                best_points, best_lines, time_limit=remaining, start_time=time.time()
            )
            if best_inc_ls > best_inc:
                best_inc = best_inc_ls
                best_lines = best_lines_ls
        except Exception:
            pass
    
    # Fallback
    if best_points is None or best_lines is None:
        pts, lns, _ = _try_grid_config(9, 6)
        best_points = pts
        best_lines = lns
    
    # Ensure integer dtype
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    return best_points, best_lines
