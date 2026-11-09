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
    For maximum total incidences, pick the top max_lines by individual incidence count.
    """
    pts = np.asarray(points, dtype=np.int64)
    n = len(pts)
    
    # Generate all unique lines determined by pairs of points
    line_map = {}  # normalized line -> set of point indices on the line
    
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
    
    # Count incidences for each line
    line_data = []
    for norm, indices in line_map.items():
        cnt = len(indices)
        if cnt > 0:
            line_data.append((cnt, norm))
    
    # Sort by count descending
    line_data.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top max_lines
    selected = line_data[:max_lines]
    lines_arr = np.array([ln for _, ln in selected], dtype=np.int64)
    
    return lines_arr

def _local_search_points(points, lines, max_iters=5000, deadline=None):
    """
    Local search: try moving individual points to new positions to increase total incidences.
    """
    pts = points.copy()
    lns = lines.copy()
    n = len(pts)
    current_score = _count_incidences(pts, lns)
    
    # Precompute line coefficients
    A = lns[:, 0]
    B = lns[:, 1]
    C = lns[:, 2]
    
    # For each point, compute which lines it's on
    def point_on_line_counts(p):
        """Return count of lines that point p lies on."""
        val = A * p[0] + B * p[1] + C
        return int(np.sum(val == 0))
    
    best_pts = pts.copy()
    best_score = current_score
    
    # Try perturbing each point
    for iter in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try moving point i to nearby positions
            x, y = pts[i]
            # Try a few random offsets
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    # Check if new position is already occupied
                    if any(pts[k, 0] == new_x and pts[k, 1] == new_y for k in range(n) if k != i):
                        continue
                    
                    old_score_i = point_on_line_counts(pts[i])
                    new_score_i = point_on_line_counts(np.array([new_x, new_y]))
                    
                    # But we also need to check if other points' line memberships change
                    # This is complex, so just compute full score for candidate
                    old_pt = pts[i].copy()
                    pts[i] = [new_x, new_y]
                    new_score = _count_incidences(pts, lns)
                    pts[i] = old_pt
                    
                    if new_score > best_score:
                        pts[i] = [new_x, new_y]
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return pts, best_score

def _build_grid(nx, ny):
    """Build a nx * ny grid of points. If nx*ny > 54, trim to 54."""
    xs = np.arange(nx)
    ys = np.arange(ny)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    if len(points) > 54:
        # Remove points from the end (highest y, highest x)
        points = points[:54]
    elif len(points) < 54:
        return None, None
    
    if len(points) != 54:
        return None, None
    
    lines = _find_optimal_lines(points, 27)
    return points, lines

def _build_perturbed_grid(nx, ny, seed=42):
    """Build a grid with some points perturbed to break symmetry and potentially increase incidences."""
    np.random.seed(seed)
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
    
    # Perturb some points
    for _ in range(20):
        i = np.random.randint(0, len(points))
        points[i, 0] += np.random.choice([-1, 0, 1])
        points[i, 1] += np.random.choice([-1, 0, 1])
    
    # Ensure no duplicates
    seen = set()
    unique_pts = []
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t not in seen:
            seen.add(t)
            unique_pts.append(p)
    
    if len(unique_pts) < 54:
        # Fill with new random points
        for _ in range(54 - len(unique_pts)):
            for attempt in range(100):
                x = np.random.randint(0, nx + 5)
                y = np.random.randint(0, ny + 5)
                if (x, y) not in seen:
                    seen.add((x, y))
                    unique_pts.append([x, y])
                    break
    
    points = np.array(unique_pts[:54], dtype=np.int64)
    lines = _find_optimal_lines(points, 27)
    return points, lines

def _build_pencil(n):
    """
    Pencil construction: n points on a line, plus 2 lines each passing through 2 points.
    Wait, the standard Szemeredi-Trotter example for n points, n lines is often:
    n points on x-axis: (0,0), (1,0), ..., (n-1,0).
    Lines: y=0 (n incidences).
    Plus lines through pairs?
    
    Let's stick to the grid which is known to be strong.
    """
    pass

def _build_szemeredi_trotter():
    """
    The classic Szemeredi-Trotter construction that achieves ~ c n^(4/3) incidences.
    For n=54, n^(4/3) = 54^1.333...
    54^(1/3) ~ 3.78.
    54 * 3.78 ~ 204.
    
    The construction:
    Let a = floor(n^(1/3)).
    Take a x a grid of "supernodes".
    Each supernode is a cluster of ~ n^(2/3) points?
    No, the standard construction uses a grid of size a x a.
    Points: a^2 points? No, we need n points.
    
    Let's use the grid approach with various aspect ratios.
    6x9 is 54.
    3x18 is 54.
    2x27 is 54.
    1x54 is 54.
    
    Also, we can try non-rectangular grids or perturbed grids.
    But first, let's see if a different aspect ratio helps.
    6x9: width 6, height 9.
    3x18: width 3, height 18.
    
    For 3x18:
    Horizontal lines: 18 lines, each with 3 points. Total 18*3 = 54.
    Vertical lines: 3 lines, each with 18 points. Total 3*18 = 54.
    Diagonals:
    Slope 1: (0,0)-(1,1)-(2,2). Length 3. How many?
    In a 3x18 grid, slope 1 diagonals have length at most 3.
    Number of such diagonals: 18-3+1 = 16 diagonals of length 3?
    Actually, diagonals of slope 1 in W x H grid.
    Length L = min(W, H, ...). Here W=3, so max length 3.
    There are W+H-1 diagonals of slope 1.
    The middle ones have length 3.
    Number of diagonals with length 3: H - W + 1 = 18 - 3 + 1 = 16.
    So 16 diagonals with 3 points. Total 16*3 = 48.
    Other diagonals have length 1 or 2.
    Slope -1: same, 16 diagonals with 3 points. Total 48.
    
    So top lines for 3x18:
    3 vertical (18 pts each) -> 54
    18 horizontal (3 pts each) -> 54
    16 diag slope 1 (3 pts each) -> 48
    16 diag slope -1 (3 pts each) -> 48
    Total top 27:
    3 vert (18) + 18 horz (3) + 6 diag (3) = 3+18+6 = 27 lines.
    Incidences: 3*18 + 18*3 + 6*3 = 54 + 54 + 18 = 126.
    
    Compare 6x9:
    6 vertical (9 pts) -> 54
    9 horizontal (6 pts) -> 54
    Diagonals slope 1: max length 6.
    Number of diagonals length 6: 9-6+1 = 4.
    Length 5: 2*(6-1)? No.
    In 6x9 grid, slope 1 diagonals:
    Lengths: 1, 2, 3, 4, 5, 6, 6, 6, 6, 5, 4, 3, 2, 1.
    Wait, H=9, W=6.
    Diagonals count: 6+9-1 = 14.
    Lengths: 1, 2, 3, 4, 5, 6, 6, 6, 6, 5, 4, 3, 2, 1.
    Sum of lengths = 54 (each point on exactly one slope 1 diagonal).
    Max 6 lines? No, only 4 lines of length 6.
    So top slope 1 diagonals: 4 lines of length 6, 2 lines of length 5.
    Total slope 1 incidences if we pick top 6: 4*6 + 2*5 = 24+10=34.
    
    Slope -1: same, 34.
    
    Other slopes?
    Slope 2: (0,0)-(1,2)-(2,4)-(3,6)-(4,8). Length 5.
    In 6x9, slope 2 max length: min(6, ceil(9/2)) = min(6,5) = 5.
    Number of such lines?
    (0,0), (1,2), (2,4), (3,6), (4,8).
    (1,0), (2,2), (3,4), (4,6), (5,8).
    (0,1), (1,3), (2,5), (3,7), (4,9) - (4,9) out of bounds? y<9, so y=8 max.
    (0,1), (1,3), (2,5), (3,7). Length 4.
    It gets complex.
    
    Let's rely on the code to compute the exact top 27 for each grid.
    """
    pass

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
        (4, 14), # 56 points, trim 2
        (5, 11), # 55 points, trim 1
        (11, 5),
        (14, 4),
    ]
    
    # First pass: find best grid
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
    
    # Second pass: try perturbed grids
    if time.time() < deadline - 100:
        for seed in [42, 123, 456, 789, 321]:
            if time.time() > deadline - 50:
                break
            for nx, ny in [(6, 9), (9, 6), (3, 18)]:
                if time.time() > deadline - 50:
                    break
                pts, lns = _build_perturbed_grid(nx, ny, seed)
                if pts is None:
                    continue
                if not _is_valid(pts, lns):
                    continue
                score = _count_incidences(pts, lns)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
                    best_lines = lns.copy()
    
    # Third pass: local search on best configuration
    if best_pts is not None and time.time() < deadline - 100:
        # Use remaining time for local search
        remaining = deadline - time.time() - 30  # 30s safety margin
        max_iters = min(10000, int(remaining * 10))
        if max_iters > 100:
            improved_pts, improved_score = _local_search_points(
                best_pts, best_lines, max_iters=max_iters, deadline=deadline - 30
            )
            if improved_score > best_score:
                best_score = improved_score
                best_pts = improved_pts.copy()
                # Re-optimize lines for the improved points
                best_lines = _find_optimal_lines(best_pts, 27)
                best_score = _count_incidences(best_pts, best_lines)
    
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
    
    return best_pts, best_lines
