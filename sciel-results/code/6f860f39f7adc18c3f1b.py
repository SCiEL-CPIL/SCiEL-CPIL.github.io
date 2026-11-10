import numpy as np
from math import gcd
from collections import Counter

def _normalize_line(A, B, C):
    """Normalize (A, B, C) to canonical form: gcd-reduced, first nonzero positive."""
    g = 0
    for v in (abs(A), abs(B), abs(C)):
        if v > 0:
            g = v if g == 0 else gcd(g, v)
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    """Compute total incidences exactly in int64."""
    total = 0
    for p in points:
        x, y = int(p[0]), int(p[1])
        for l in lines:
            A, B, C = int(l[0]), int(l[1]), int(l[2])
            if A * x + B * y + C == 0:
                total += 1
    return total

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Construction: 8x7 grid (56 points) with 2 strategically removed points.
    - 7 horizontal lines (y=1..7): 7*8 = 56 incidences (minus removed)
    - 8 vertical lines (x=1..8): 8*7 = 56 incidences (minus removed)
    - 12 mixed-slope lines selected greedily for maximum coverage
    
    The 8x7 grid allows mixed lines (e.g., diagonals) to cover up to 7 points,
    compared to 6 in a 6x9 grid.
    """
    # Start with 8x7 grid: x in 1..8, y in 1..7
    xs = np.arange(1, 9)
    ys = np.arange(1, 8)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Remove 2 points that are NOT on the main 7-point diagonals
    # Main diagonals: y-x = 0, y-x = -1, y+x = 8, y+x = 9 (for 8x7 grid)
    # Points to remove: (1,3) and (1,4) - these are on y-x=2 and y-x=3, not main diagonals
    remove_set = {(1, 3), (1, 4)}
    
    points_list = []
    for p in all_points:
        if (int(p[0]), int(p[1])) not in remove_set:
            points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    lines = []
    existing_norms = set()
    
    # Horizontal lines: y = c -> (0, 1, -c)
    for c in range(1, 8):
        norm = _normalize_line(0, 1, -c)
        if norm not in existing_norms:
            lines.append(norm)
            existing_norms.add(norm)
    
    # Vertical lines: x = c -> (1, 0, -c)
    for c in range(1, 9):
        norm = _normalize_line(1, 0, -c)
        if norm not in existing_norms:
            lines.append(norm)
            existing_norms.add(norm)
    
    # Generate candidate mixed-slope lines
    # For each (A, B) pair, compute C values for all points and count occurrences
    coeff_pairs = [
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2),
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 3), (1, -3), (-1, 3), (-1, -3),
        (3, 1), (3, -1), (-3, 1), (-3, -1),
        (1, 4), (1, -4), (-1, 4), (-1, -4),
        (4, 1), (4, -1), (-4, 1), (-4, -1),
        (2, 3), (2, -3), (-2, 3), (-2, -3),
        (3, 2), (3, -2), (-3, 2), (-3, -2),
    ]
    
    candidates = []
    for A, B in coeff_pairs:
        if A == 0 and B == 0:
            continue
        c_vals = -(A * points[:, 0] + B * points[:, 1])
        counts = Counter(c_vals.tolist())
        for C, count in counts.items():
            if count >= 2:
                norm = _normalize_line(A, B, C)
                if norm not in existing_norms:
                    candidates.append((count, norm))
    
    # Sort by count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Greedily add 12 more lines
    added = 0
    for count, norm in candidates:
        if added == 12:
            break
        if norm not in existing_norms:
            lines.append(norm)
            existing_norms.add(norm)
            added += 1
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
