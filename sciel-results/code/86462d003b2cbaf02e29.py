import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    """Return a canonical integer triple for the line A*x + B*y + C = 0."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    # Make the first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _build_grid(m, n, offset_x=0, offset_y=0):
    """Build an m x n grid of integer points."""
    xs = np.arange(offset_x + 1, offset_x + m + 1, dtype=np.int64)
    ys = np.arange(offset_y + 1, offset_y + n + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _get_all_lines(pts):
    """Get all lines through >=2 points with their incidence counts."""
    n = len(pts)
    line_set = {}
    
    for i in range(n):
        xi, yi = int(pts[i, 0]), int(pts[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(pts[j, 0]), int(pts[j, 1])
            A = yi - yj
            B = xj - xi
            C = -(xi * yj - xj * yi)
            key = _normalize(A, B, C)
            if key not in line_set:
                line_set[key] = set()
            line_set[key].add(i)
            line_set[key].add(j)
    
    # Count incidences
    line_counts = {}
    for key, point_indices in line_set.items():
        line_counts[key] = len(point_indices)
    
    return line_counts


def _select_top_lines(line_counts, num_lines=27):
    """Select the top num_lines lines by incidence count."""
    candidates = [(cnt, key) for key, cnt in line_counts.items() if cnt >= 2]
    candidates.sort(key=lambda t: (-t[0], t[1]))
    
    selected = []
    seen = set()
    for cnt, key in candidates:
        if key in seen:
            continue
        seen.add(key)
        selected.append((cnt, key))
        if len(selected) == num_lines:
            break
    
    return selected


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    
    Strategy: Try multiple point configurations (different grid shapes and
    offsets) and select the one that gives the highest sum of top-27 line
    incidence counts.
    """
    t_start = time.time()
    deadline = 600.0
    margin = 30.0
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Try various grid configurations
    configs = [
        (6, 9, 0, 0),
        (9, 6, 0, 0),
        (3, 18, 0, 0),
        (18, 3, 0, 0),
        (2, 27, 0, 0),
        (27, 2, 0, 0),
        (6, 9, 1, 0),
        (6, 9, 0, 1),
    ]
    
    for m, n, ox, oy in configs:
        if time.time() > deadline - margin:
            break
        
        pts = _build_grid(m, n, ox, oy)
        line_counts = _get_all_lines(pts)
        selected = _select_top_lines(line_counts, 27)
        
        if len(selected) < 27:
            continue
        
        score = sum(cnt for cnt, _ in selected)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = np.array([key for _, key in selected], dtype=np.int64)
    
    # If we haven't found anything, use default 6x9 grid
    if best_points is None:
        best_points = _build_grid(6, 9, 0, 0)
        line_counts = _get_all_lines(best_points)
        selected = _select_top_lines(line_counts, 27)
        if len(selected) < 27:
            # Fallback
            k = 3
            xs2 = np.arange(1, k + 1)
            ys2 = np.arange(1, 2 * k * k + 1)
            px2, py2 = np.meshgrid(xs2, ys2, indexing="ij")
            best_points = np.stack([px2.ravel(), py2.ravel()], axis=1).astype(np.int64)
            rows = []
            for a in range(1, k + 1):
                for b in range(1, k * k + 1):
                    rows.append((a, -1, b))
            best_lines = np.array(rows, dtype=np.int64)
        else:
            best_lines = np.array([key for _, key in selected], dtype=np.int64)
    
    return best_points, best_lines
