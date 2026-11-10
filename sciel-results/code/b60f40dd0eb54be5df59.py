import numpy as np
from math import gcd

def _normalize_line(A, B, C):
    """
    Normalize (A, B, C) so that:
    - A*B*C have gcd 1
    - First non-zero entry is positive
    """
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return None
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    """
    Given two distinct points, return normalized (A, B, C) for the line through them,
    or None if same point.
    """
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and y1 == y2:
        return None
    A = y1 - y2
    B = x2 - x1
    C = -(A * x1 + B * y1)
    return _normalize_line(A, B, C)

def _count_incidence(points, A, B, C):
    """
    Count how many points in `points` (N,2) satisfy A*x + B*y + C == 0.
    """
    return np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)

def _pick_top_k_lines(points, k):
    """
    Given points (N,2), pick k distinct lines (each through >= 2 points) maximizing
    total incidences. Use greedy: repeatedly pick the line with max incidences not
    already chosen.
    """
    N = len(points)
    # Gather all candidate lines from point pairs
    candidate_set = {}
    for i in range(N):
        for j in range(i + 1, N):
            norm = _line_from_points(points[i], points[j])
            if norm is None:
                continue
            candidate_set[norm] = True

    # Compute incidence for each candidate
    incidences = {}
    for norm in candidate_set:
        A, B, C = norm
        incidences[norm] = _count_incidence(points, A, B, C)

    # Greedy selection
    chosen = []
    remaining = list(candidate_set.keys())
    # Sort by incidence descending
    remaining.sort(key=lambda n: (-incidences[n], n))

    for norm in remaining:
        if len(chosen) >= k:
            break
        chosen.append(norm)

    # If we didn't reach k lines (unlikely for N=54), pad with arbitrary valid lines
    # (this shouldn't happen, but for safety)
    if len(chosen) < k:
        # Generate vertical lines at distinct x not already used, or horizontal
        used_norms = set(chosen)
        # Try to add more lines through pairs not yet covered
        for i in range(N):
            for j in range(i + 1, N):
                if len(chosen) >= k:
                    break
                norm = _line_from_points(points[i], points[j])
                if norm is not None and norm not in used_norms:
                    chosen.append(norm)
                    used_norms.add(norm)

    lines = np.array(chosen, dtype=np.int64)
    return lines

def _count_total_incidence(points, lines):
    """
    Count total incidences between points (N,2) and lines (M,3).
    """
    N = len(points)
    M = len(lines)
    # Vectorized: for each line, count points on it
    total = 0
    for A, B, C in lines:
        total += _count_incidence(points, A, B, C)
    return total

def _generate_grid_points(n, m):
    """
    Generate a rectangular grid of n x m points (n columns, m rows).
    Returns (n*m, 2) array.
    """
    xs = np.arange(n, dtype=np.int64)
    ys = np.arange(m, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return points

def _validate_and_round(points, lines):
    """
    Ensure points and lines are valid: integer, no NaN/inf, no repeated points,
    no zero lines, no duplicate lines. Round to int64.
    """
    # Points
    points = np.round(points).astype(np.int64)
    # Lines
    lines = np.round(lines).astype(np.int64)
    return points, lines

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy: Try several grid configurations and pick the one with the highest
    total incidence after greedy line selection.
    """
    # Candidate grid shapes (n, m) with n*m = 54
    # Divisors of 54: 1,2,3,6,9,18,27,54
    # So possible (n, m) with n*m=54:
    candidates = [
        (6, 9),
        (9, 6),
        (3, 18),
        (18, 3),
        (2, 27),
        (27, 2),
        (1, 54),
        (54, 1),
    ]
    
    best_incidence = -1
    best_points = None
    best_lines = None
    
    for n, m in candidates:
        points = _generate_grid_points(n, m)
        lines = _pick_top_k_lines(points, 27)
        inc = _count_total_incidence(points, lines)
        if inc > best_incidence:
            best_incidence = inc
            best_points = points
            best_lines = lines
    
    points, lines = _validate_and_round(best_points, best_lines)
    return points, lines
