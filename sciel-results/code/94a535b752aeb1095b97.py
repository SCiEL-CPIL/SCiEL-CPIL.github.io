import numpy as np
import time
from math import gcd
from itertools import product


def _count_incidences(points, lines):
    """Count point-line incidences exactly using int64 arithmetic."""
    # points: (n, 2) int64, lines: (m, 3) int64
    # For each line (A, B, C) and point (x, y): A*x + B*y + C == 0
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(m):
        A, B, C = lines[i]
        for j in range(n):
            x, y = points[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _grid_config(a, b):
    """Generate a×b grid of points and candidate lines."""
    # Points: {0,...,a-1} x {0,...,b-1}
    xs = np.arange(a, dtype=np.int64)
    ys = np.arange(b, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1)
    
    # Candidate lines: rows, columns, diagonals
    candidates = []
    
    # Horizontal lines: y = c -> (0, 1, -c)
    for c in range(b):
        candidates.append((0, 1, -c))
    
    # Vertical lines: x = c -> (1, 0, -c)
    for c in range(a):
        candidates.append((1, 0, -c))
    
    # Diagonal lines with slope 1: x - y = c -> (1, -1, -c)
    for c in range(-(b-1), a):
        candidates.append((1, -1, -c))
    
    # Diagonal lines with slope -1: x + y = c -> (1, 1, -c)
    for c in range(0, a+b-1):
        candidates.append((1, 1, -c))
    
    # Lines with slope 2: y = 2x + c -> (2, -1, c)
    for c in range(-2*(a-1), b):
        candidates.append((2, -1, c))
    
    # Lines with slope 1/2: y = x/2 + c -> (1, -2, 2c)
    for c in range(0, (b+1)//2 + 1):
        candidates.append((1, -2, 2*c))
    
    return points, candidates


def _select_best_lines(points, candidates, m=27):
    """Select m lines with highest incidence counts, avoiding duplicates."""
    n = len(points)
    
    # Compute incidence count for each candidate line
    line_info = []
    seen = set()
    for (A, B, C) in candidates:
        # Normalize: divide by gcd, make first nonzero positive
        g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
        if g == 0:
            continue
        A_n, B_n, C_n = A // g, B // g, C // g
        # Make first nonzero positive
        if A_n < 0 or (A_n == 0 and B_n < 0) or (A_n == 0 and B_n == 0 and C_n < 0):
            A_n, B_n, C_n = -A_n, -B_n, -C_n
        key = (A_n, B_n, C_n)
        if key in seen:
            continue
        seen.add(key)
        
        # Count incidences
        count = 0
        for (x, y) in points:
            if A * x + B * y + C == 0:
                count += 1
        line_info.append((count, A, B, C))
    
    # Sort by count descending, take top m
    line_info.sort(key=lambda x: -x[0])
    selected = line_info[:m]
    
    lines = np.array([(A, B, C) for (_, A, B, C) in selected], dtype=np.int64)
    total = sum(count for (count, _, _, _) in selected)
    
    return points, lines, total


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline with safety margin
    
    # Factorizations of 54: 1x54, 2x27, 3x18, 6x9
    factorizations = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    
    best_points = None
    best_lines = None
    best_total = -1
    
    for (a, b) in factorizations:
        if time.time() > deadline - 60:
            break
        
        points, candidates = _grid_config(a, b)
        points, lines, total = _select_best_lines(points, candidates, 27)
        
        if total > best_total:
            best_total = total
            best_points = points.copy()
            best_lines = lines.copy()
    
    # If we have time, try some local improvements
    if best_points is not None and time.time() < deadline - 120:
        # Try shifting the grid
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if time.time() > deadline - 60:
                    break
                shifted_points = best_points + np.array([dx, dy], dtype=np.int64)
                # Re-select lines for shifted points
                _, candidates = _grid_config(len(set(shifted_points[:, 0])), len(set(shifted_points[:, 1])))
                # This is getting complex; skip for now
                pass
    
    # Ensure valid output
    if best_points is None or best_lines is None:
        # Fallback to baseline
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    return best_points, best_lines
