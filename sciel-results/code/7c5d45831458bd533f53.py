import numpy as np
import time
import math
from itertools import product, combinations
from math import gcd

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) by dividing by gcd and making first nonzero positive."""
    A, B, C = int(A), int(B), int(C)
    if A == 0 and B == 0:
        return None  # Invalid
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
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
    # Check no NaN/inf (int64 arrays can't have those, but check finite)
    if not np.all(np.isfinite(pts.astype(np.float64))):
        return False
    if not np.all(np.isfinite(lns.astype(np.float64))):
        return False
    # No repeated points
    seen_pts = set()
    for p in pts:
        t = (int(p[0]), int(p[1]))
        if t in seen_pts:
            return False
        seen_pts.add(t)
    # No invalid or duplicate lines
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

def _build_6x9():
    """Classical Szemeredi-Trotter k=3 construction: 54 points, 27 lines, 81 incidences."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            lines.append((a, -1, b))
    lines_arr = np.array(lines, dtype=np.int64)
    return points, lines_arr

def _build_grid6x9():
    """6x9 grid with top-27 lines by incidence count."""
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    candidates = []
    # Vertical
    for c in range(6):
        candidates.append((1, 0, -c))
    # Horizontal
    for r in range(9):
        candidates.append((0, 1, -r))
    # Diagonal y - x = d
    for d in range(-5, 6):
        candidates.append((-1, 1, d))
    # Anti-diagonal x + y = s
    for s in range(0, 15):
        candidates.append((1, 1, -s))

    # Score each candidate
    scores = []
    for idx, (A, B, C) in enumerate(candidates):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [candidates[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    return points, lines_arr

def _build_product_grid():
    """
    Product construction: P = A x B, L = lines through A x B.
    Choose A with |A|=a, B with |B|=b, a*b=54.
    Lines: for each pair (a1,a2) in A and (b1,b2) in B, line through (a1,b1),(a2,b2).
    That's C(a,2)*C(b,2) lines, each with exactly 2 incidences.
    We need exactly 27 lines.
    
    Try a=6, b=9: C(6,2)*C(9,2) = 15*36 = 540 lines, pick top 27.
    Each line has 2 incidences, so 27*2 = 54. Worse than 81.
    
    Alternative: use lines that pass through more points.
    In a 6x9 grid, lines can pass through multiple points.
    Let's just use the grid6x9 approach but with better point selection.
    """
    return _build_grid6x9()

def _build_pencil_grid():
    """
    Pencil construction: points near a line, with many lines through pairs.
    Place 27 points on a line (the x-axis) and 27 points scattered.
    Lines: for each pair of scattered points, the line through them.
    But that gives C(27,2) = 351 lines, each with 2 incidences among scattered points,
    plus potentially hitting axis points.
    
    Better: 13 points on x-axis, 41 elsewhere? No, let's think differently.
    
    Actually, let's try: 27 points on x-axis at x=1..27, y=0.
    27 points at (i, i) for i=1..27? No, those are collinear too.
    
    Let's do: 27 points on x-axis: (1,0), (2,0), ..., (27,0)
    27 points on y-axis: (0,1), (0,2), ..., (0,27)
    Lines: x + y = c for various c. Each line x+y=c hits one x-axis point (c,0) 
    and one y-axis point (0,c) if 1<=c<=27. That's 2 incidences per line.
    27 lines -> 54 incidences. Still less than 81.
    
    Let's stick with the grid approach and try to optimize.
    """
    return _build_grid6x9()

def _build_optimized_grid():
    """
    Try different grid sizes and line selections to maximize incidences.
    """
    best_pts = None
    best_lns = None
    best_inc = -1
    
    # Try 6x9, 9x6, 3x18, 18x3, 2x27, 27x2
    for (a, b) in [(6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2)]:
        xs = np.arange(a)
        ys = np.arange(b)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Generate candidate lines
        candidates = []
        # Vertical
        for c in range(a):
            candidates.append((1, 0, -c))
        # Horizontal
        for r in range(b):
            candidates.append((0, 1, -r))
        # Diagonals y - x = d
        for d in range(-(a-1), a):
            candidates.append((-1, 1, d))
        # Anti-diagonals x + y = s
        for s in range(0, a + b - 1):
            candidates.append((1, 1, -s))
        # Steeper slopes
        for slope in [2, 3, -2, -3]:
            for c in range(-b, b):
                candidates.append((slope, -1, c))
        
        # Score and pick top 27
        scores = []
        for idx, (A, B, C) in enumerate(candidates):
            val = A * points[:, 0] + B * points[:, 1] + C
            cnt = int(np.sum(val == 0))
            scores.append((cnt, idx))
        scores.sort(reverse=True)
        top = [candidates[idx] for _, idx in scores[:27]]
        lines_arr = np.array(top, dtype=np.int64)
        
        if _is_valid(points, lines_arr):
            inc = _count_incidences(points, lines_arr)
            if inc > best_inc:
                best_inc = inc
                best_pts, best_lns = points, lines_arr
    
    if best_pts is not None:
        return best_pts, best_lns
    return _build_6x9()

def _build_pencil_plus():
    """
    Pencil construction: n/2 points on a line, n/2 points off.
    Lines through pairs of off-line points that also hit the main line.
    """
    # 27 points on x-axis
    axis_pts = [(i, 0) for i in range(1, 28)]
    # 27 points above
    off_pts = [(i, i) for i in range(1, 28)]
    
    all_pts = axis_pts + off_pts
    points = np.array(all_pts, dtype=np.int64)
    
    # Lines through pairs of off_pts
    lines = []
    for i in range(27):
        for j in range(i+1, 27):
            x1, y1 = off_pts[i]
            x2, y2 = off_pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm is not None:
                lines.append(norm)
    
    # Deduplicate and score
    seen = set()
    unique_lines = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique_lines.append(l)
    
    # Score each line
    scores = []
    for idx, (A, B, C) in enumerate(unique_lines):
        val = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(val == 0))
        scores.append((cnt, idx))
    scores.sort(reverse=True)
    top = [unique_lines[idx] for _, idx in scores[:27]]
    lines_arr = np.array(top, dtype=np.int64)
    
    if _is_valid(points, lines_arr):
        return points, lines_arr
    return _build_6x9()

def _build_perturbed_grid():
    """
    Try slightly perturbed grid positions to break symmetry and allow more lines.
    """
    base_pts, base_lns = _build_grid6x9()
    return base_pts, base_lns

def szemeredi_trotter_incidences():
    deadline = time.time() + 580.0
    
    candidates = [
        ("sz6x18", _build_6x9),
        ("grid6x9", _build_grid6x9),
        ("opt_grid", _build_optimized_grid),
        ("pencil", _build_pencil_plus),
    ]
    
    best_pts = None
    best_lns = None
    best_inc = -1
    
    for name, fn in candidates:
        if time.time() > deadline:
            break
        try:
            pts, lns = fn()
            if _is_valid(pts, lns):
                inc = _count_incidences(pts, lns)
                if inc > best_inc:
                    best_inc = inc
                    best_pts, best_lns = pts, lns
        except Exception:
            continue
    
    if best_pts is None:
        pts, lns = _build_6x9()
        return pts, lns
    
    return best_pts, best_lns
