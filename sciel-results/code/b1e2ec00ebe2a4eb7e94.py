# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd


def _normalize_line(A, B, C):
    """Reduce (A,B,C) by gcd and make first nonzero entry positive."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(int(A)), gcd(abs(int(B)), abs(int(C))))
    if g > 0:
        A //= g
        B //= g
        C //= g
    # Make first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidents(points, lines):
    """Count exact int64 incidences."""
    total = 0
    for A, B, C in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.count_nonzero(vals == 0))
    return total


def _line_from_points(p1, p2):
    """Compute normalized line (A,B,C) through two points."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    return _normalize_line(A, B, C)


def _build_6x9_grid():
    """
    Build a 6x9 grid and select 27 lines to maximize incidences.
    
    Points: 6 columns (x=1..6) x 9 rows (y=1..9) = 54 points.
    
    Lines selected:
    - 6 vertical lines (x=c), each with 9 points: 54 incidences
    - 8 diagonal lines with slope 1 (y-x=c) having 6 points: 48 incidences
    - 8 diagonal lines with slope -1 (y+x=c) having 6 points: 48 incidences
    - 5 more lines (slope 2 or other) with ~4 points each: ~20 incidences
    Total: ~170 incidences
    """
    xs = np.arange(1, 7)  # 1..6
    ys = np.arange(1, 10)  # 1..9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    seen = set()
    
    # Add vertical lines: x = c for c in 1..6
    for c in range(1, 7):
        line = (1, 0, -c)
        if line not in seen:
            lines.append(line)
            seen.add(line)
    
    # Add diagonal lines with slope 1: y - x = c
    # For a 6x9 grid, y-x ranges from (1-6)=-5 to (9-1)=8
    # Lines with 6 points: need to check which c values give 6 points
    # y = x + c, where 1<=x<=6, 1<=y<=9
    # 1 <= x+c <= 9 => 1-c <= x <= 9-c
    # Also 1 <= x <= 6
    # So max(1, 1-c) <= x <= min(6, 9-c)
    # Number of points = min(6, 9-c) - max(1, 1-c) + 1 if valid
    
    slope1_lines = []
    for c in range(-5, 9):
        x_lo = max(1, 1 - c)
        x_hi = min(6, 9 - c)
        if x_lo <= x_hi:
            n_pts = x_hi - x_lo + 1
            slope1_lines.append((c, n_pts))
    
    # Sort by number of points descending, then by c
    slope1_lines.sort(key=lambda x: (-x[1], x[0]))
    
    # Add top 8 slope-1 lines
    count = 0
    for c, n_pts in slope1_lines:
        if count >= 8:
            break
        # y - x = c => -x + y - c = 0 => A=-1, B=1, C=-c
        line = _normalize_line(-1, 1, -c)
        if line not in seen:
            lines.append(line)
            seen.add(line)
            count += 1
    
    # Add diagonal lines with slope -1: y + x = c
    # y = c - x, where 1<=x<=6, 1<=y<=9
    # 1 <= c-x <= 9 => c-9 <= x <= c-1
    # Also 1 <= x <= 6
    # So max(1, c-9) <= x <= min(6, c-1)
    
    slope_neg1_lines = []
    for c in range(2, 16):  # min c=1+1=2, max c=6+9=15
        x_lo = max(1, c - 9)
        x_hi = min(6, c - 1)
        if x_lo <= x_hi:
            n_pts = x_hi - x_lo + 1
            slope_neg1_lines.append((c, n_pts))
    
    slope_neg1_lines.sort(key=lambda x: (-x[1], x[0]))
    
    count = 0
    for c, n_pts in slope_neg1_lines:
        if count >= 8:
            break
        # y + x = c => x + y - c = 0 => A=1, B=1, C=-c
        line = _normalize_line(1, 1, -c)
        if line not in seen:
            lines.append(line)
            seen.add(line)
            count += 1
    
    # We have 6 + 8 + 8 = 22 lines. Need 5 more.
    # Try slope 2: y = 2x + c => -2x + y - c = 0
    # For x in 1..6, y in 1..9
    # 1 <= 2x+c <= 9 => (1-c)/2 <= x <= (9-c)/2
    # Also 1 <= x <= 6
    
    slope2_lines = []
    for c in range(-11, 10):  # rough range
        # For each x in 1..6, check if y=2x+c is in 1..9
        pts_on_line = []
        for x in range(1, 7):
            y = 2 * x + c
            if 1 <= y <= 9:
                pts_on_line.append((x, y))
        if len(pts_on_line) >= 2:
            slope2_lines.append((c, len(pts_on_line)))
    
    slope2_lines.sort(key=lambda x: (-x[1], x[0]))
    
    count = 0
    for c, n_pts in slope2_lines:
        if count >= 5:
            break
        if len(lines) >= 27:
            break
        # y = 2x + c => -2x + y - c = 0
        line = _normalize_line(-2, 1, -c)
        if line not in seen:
            lines.append(line)
            seen.add(line)
            count += 1
    
    # If we still don't have 27, add more lines from remaining candidates
    if len(lines) < 27:
        # Try other slopes
        for slope in [3, -2, -3, 4, -4, 1/2]:
            if len(lines) >= 27:
                break
            # This is getting complicated; let's just iterate over all pairs
            # and add lines until we have 27
            pass
    
    # Fallback: iterate over all pairs and add lines until we have 27
    while len(lines) < 27:
        added = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if len(lines) >= 27:
                    break
                line = _line_from_points(points[i], points[j])
                if line not in seen:
                    lines.append(line)
                    seen.add(line)
                    added = True
            if len(lines) >= 27:
                break
        if not added:
            break
    
    lines = lines[:27]
    lines_array = np.array(lines, dtype=np.int64)
    
    return points, lines_array


def _validate(points, lines):
    """Return True if the configuration satisfies all contract validity rules."""
    if points.shape != (54, 2):
        return False
    if lines.shape != (27, 3):
        return False
    if not (np.issubdtype(points.dtype, np.integer) and np.issubdtype(lines.dtype, np.integer)):
        return False
    # repeated points
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    seen_lines = set()
    for A, B, C in lines.tolist():
        if A == 0 and B == 0:
            return False
        c = _normalize_line(A, B, C)
        if c in seen_lines:
            return False
        seen_lines.add(c)
    return True


def _optimize(points, lines, deadline):
    """
    Local search to improve incidence count.
    Try swapping lines to improve the total.
    """
    if not _validate(points, lines):
        return points, lines
    
    best_count = _count_incidents(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Precompute for each line the set of points on it
    def get_line_points(line_idx):
        A, B, C = lines[line_idx]
        vals = A * points[:, 0] + B * points[:, 1] + C
        return np.where(vals == 0)[0]
    
    # Try single-line swaps: replace one line with a line through two points
    # that isn't already present
    existing_lines = set(map(tuple, lines.tolist()))
    
    # Generate candidate lines from all pairs of points
    candidate_lines = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            line = _line_from_points(points[i], points[j])
            key = (line[0], line[1], line[2])
            if key not in candidate_lines:
                candidate_lines[key] = line
    
    candidate_list = list(candidate_lines.values())
    
    # Local search: try replacing each line with a candidate line
    # to maximize the change in incidence count
    improved = True
    max_iters = 500
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Time check
        if time.time() > deadline - 30:
            break
        
        for old_idx in range(len(lines)):
            if time.time() > deadline - 30:
                break
            
            old_line = lines[old_idx]
            old_pts = get_line_points(old_idx)
            old_count = len(old_pts)
            
            # Find best replacement
            best_new_count = -1
            best_new_line = None
            
            # Only consider lines that would improve or maintain count
            for new_line in candidate_list:
                new_key = (new_line[0], new_line[1], new_line[2])
                if new_key == (old_line[0], old_line[1], old_line[2]):
                    continue
                if new_key in existing_lines and new_key != (old_line[0], old_line[1], old_line[2]):
                    continue
                
                # Count points on new line
                A, B, C = new_line
                vals = A * points[:, 0] + B * points[:, 1] + C
                new_pts = np.where(vals == 0)[0]
                new_count = len(new_pts)
                
                # Net change: remove old line's contribution, add new line's
                # But points on both old and new lines are counted in both
                # So we need to be careful
                old_set = set(old_pts.tolist())
                new_set = set(new_pts.tolist())
                
                # Points only on old line (lost): old_set - new_set
                # Points only on new line (gained): new_set - old_set
                # Points on both: no change
                lost = len(old_set - new_set)
                gained = len(new_set - old_set)
                net_change = gained - lost
                
                if net_change > best_new_count:
                    best_new_count = net_change
                    best_new_line = new_line
            
            if best_new_line is not None and best_new_count > 0:
                # Perform swap
                old_key = (old_line[0], old_line[1], old_line[2])
                existing_lines.discard(old_key)
                new_key = (best_new_line[0], best_new_line[1], best_new_line[2])
                existing_lines.add(new_key)
                
                lines[old_idx] = best_new_line
                best_count += best_new_count
                improved = True
                
                if _count_incidents(points, lines) > best_count:
                    best_count = _count_incidents(points, lines)
                    best_points = points.copy()
                    best_lines = lines.copy()
    
    return best_points, best_lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization for n=54 points, m=27 lines.
    
    Uses a 6x9 grid construction with carefully selected lines, then local search.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # 30s safety margin
    
    # Build initial configuration
    points, lines = _build_6x9_grid()
    
    # Validate
    if not _validate(points, lines):
        # Fallback to simpler construction
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = []
        for c in range(1, 7):
            lines_list.append((1, 0, -c))
        for c in range(-5, 9):
            line = _normalize_line(-1, 1, -c)
            if len(lines_list) < 27 and line not in [tuple(l) for l in lines_list]:
                lines_list.append(line)
        lines = np.array(lines_list[:27], dtype=np.int64)
    
    # Local search if time permits
    if time.time() < deadline - 60:
        points, lines = _optimize(points, lines, deadline)
    
    # Final validation
    if not _validate(points, lines):
        # Return the initial construction which should be valid
        points, lines = _build_6x9_grid()
    
    return points, lines


# EVOLVE-BLOCK-END
