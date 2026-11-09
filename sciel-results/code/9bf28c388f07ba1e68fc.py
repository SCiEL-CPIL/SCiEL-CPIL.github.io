import numpy as np
import time
import math
from itertools import combinations
from math import gcd

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

def _build_optimal_grid_6x9():
    """
    Builds a 6x9 grid and selects the 27 lines with the highest number of incidences.
    """
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Find all unique lines defined by pairs of points
    line_counts = {}
    n_pts = len(points)
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            norm = _normalize_line(A, B, C)
            if norm is None:
                continue
            
            if norm not in line_counts:
                line_counts[norm] = 0
            
            # Increment count for this line
            # We need to count how many of the 54 points are on this line.
            # Doing this for every pair is O(N^3) which is 54^3 ~ 150k, acceptable.
            # But we can optimize: just check all points once per unique line found.
            # However, we are finding lines one by one.
            # Let's collect unique lines first, then count.
            pass

    # Collect unique lines
    unique_lines = list(line_counts.keys())
    
    # Count incidences for each unique line
    line_incidences = []
    for A, B, C in unique_lines:
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        line_incidences.append((cnt, (A, B, C)))
        
    # Sort by incidence count descending
    line_incidences.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 27
    top_lines = [ln for _, ln in line_incidences[:27]]
    lines_arr = np.array(top_lines, dtype=np.int64)
    
    return points, lines_arr

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    # The 6x9 grid with optimal line selection is the strongest candidate.
    try:
        pts, lns = _build_optimal_grid_6x9()
        if _is_valid(pts, lns):
            return pts, lns
    except Exception:
        pass
        
    # Fallback to a simple construction if optimal fails
    xs = np.arange(1, 4)
    ys = np.arange(1, 19)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines = []
    for a in range(1, 4):
        for b in range(1, 10):
            lines.append((a, -1, b))
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points, lines_arr
