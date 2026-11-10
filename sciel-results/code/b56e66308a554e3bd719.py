import numpy as np
import time
import math
from math import gcd
from itertools import product

def _gcd3(a, b, c):
    return gcd(gcd(abs(a), abs(b)), abs(c))

def _normalize_line(A, B, C):
    g = _gcd3(A, B, C)
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    # points: (n,2), lines: (m,3)
    # incidence if A*x + B*y + C == 0
    # points @ lines.T style
    P = points  # (n, 2)
    L = lines   # (m, 3)
    # A*x + B*y + C = 0
    vals = P[:, 0:1] * L[0:1, :] + P[:, 1:2] * L[1:2, :] + L[2:3, :]
    return int(np.sum(vals == 0))

def _validate(points, lines):
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    # Check integer-ness
    if not np.allclose(points, np.round(points), atol=1e-6):
        return False
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Check no NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    # Check repeated points
    pts_set = set()
    for p in points:
        key = (int(p[0]), int(p[1]))
        if key in pts_set:
            return False
        pts_set.add(key)
    # Check lines: A=B=0 invalid, duplicate lines
    line_set = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        norm = _normalize_line(A, B, C)
        if norm in line_set:
            return False
        line_set.add(norm)
    return True

def _grid_construction(nx, ny, num_lines, line_selection):
    """
    nx x ny grid of points. Select num_lines lines from a candidate pool.
    line_selection: list of (type, params) or a function
    """
    xs = np.arange(1, nx + 1)
    ys = np.arange(1, ny + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Build candidate lines
    candidates = []
    # Horizontal lines: y = c  ->  0*x + 1*y - c = 0
    for c in range(1, ny + 1):
        candidates.append((0, 1, -c))
    # Vertical lines: x = c  ->  1*x + 0*y - c = 0
    for c in range(1, nx + 1):
        candidates.append((1, 0, -c))
    # Diagonal lines: y = a*x + b -> a*x - y + b = 0
    # Try small slopes and intercepts
    for a in range(-3, 4):
        if a == 0:
            continue
        for b in range(-10, 11):
            candidates.append((a, -1, b))
    # Other diagonals: x = a*y + b -> x - a*y - b = 0
    for a in range(-3, 4):
        if a == 0:
            continue
        for b in range(-10, 11):
            candidates.append((1, -a, -b))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        norm = _normalize_line(*c)
        if norm and norm not in seen:
            seen.add(norm)
            unique_candidates.append(c)
    
    # Score each candidate by number of incidences
    scored = []
    for l in unique_candidates:
        A, B, C = l
        count = int(np.sum(points[:, 0] * A + points[:, 1] * B + C == 0))
        scored.append((count, l))
    
    scored.sort(key=lambda x: -x[0])
    
    # Greedily select lines maximizing total incidences
    # But lines can share points, so greedy isn't optimal. 
    # For now, just take top num_lines
    selected = [s[1] for s in scored[:num_lines]]
    lines = np.array(selected, dtype=np.int64)
    
    return points, lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    best_points = None
    best_lines = None
    best_score = 0
    
    # Try various grid dimensions
    grid_configs = [
        (3, 18), (6, 9), (9, 6), (18, 3),
        (2, 27), (27, 2), (1, 54), (54, 1),
        (4, 14), (14, 4), (5, 11), (11, 5),
    ]
    
    for nx, ny in grid_configs:
        if nx * ny != 54:
            continue
        if time.time() > deadline:
            break
        
        points, lines = _grid_construction(nx, ny, 27, None)
        if _validate(points, lines):
            score = _count_incidents(points, lines)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = lines.copy()
    
    # Also try the original Szemeredi-Trotter construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    st_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    st_lines = np.array(rows, dtype=np.int64)
    if _validate(st_points, st_lines):
        score = _count_incidents(st_points, st_lines)
        if score > best_score:
            best_score = score
            best_points = st_points.copy()
            best_lines = st_lines.copy()
    
    if best_points is None:
        # Fallback
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing='ij')
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    return best_points, best_lines
