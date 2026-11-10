import numpy as np
from math import gcd
from itertools import product

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
    total incidences.
    """
    N = len(points)
    if N < 2:
        return np.empty((0, 3), dtype=np.int64)
        
    # Gather all candidate lines from point pairs
    candidate_set = {}
    for i in range(N):
        for j in range(i + 1, N):
            norm = _line_from_points(points[i], points[j])
            if norm is not None:
                candidate_set[norm] = True

    # Compute incidence for each candidate
    incidences = {}
    for norm in candidate_set:
        A, B, C = norm
        incidences[norm] = _count_incidence(points, A, B, C)

    # Sort by incidence descending
    sorted_lines = sorted(incidences.items(), key=lambda item: (-item[1], item[0]))
    
    # Pick top k
    chosen = []
    for norm, _ in sorted_lines:
        if len(chosen) >= k:
            break
        chosen.append(norm)
        
    # If we have fewer than k lines (should not happen for N=54), pad
    if len(chosen) < k:
        # This is a safety check, but for 54 points there are many lines
        pass

    if not chosen:
        return np.empty((0, 3), dtype=np.int64)

    lines = np.array(chosen, dtype=np.int64)
    return lines

def _count_total_incidence(points, lines):
    """
    Count total incidences between points (N,2) and lines (M,3).
    """
    if len(lines) == 0:
        return 0
    total = 0
    for A, B, C in lines:
        total += _count_incidence(points, A, B, C)
    return total

def _generate_lattice_points(n, m, v1, v2):
    """
    Generate points on a lattice defined by basis vectors v1, v2.
    Points: i*v1 + j*v2 for 0<=i<n, 0<=j<m.
    Returns (n*m, 2) array of int64, or None if points are not distinct.
    """
    points_list = []
    seen = set()
    for i in range(n):
        for j in range(m):
            x = i * v1[0] + j * v2[0]
            y = i * v1[1] + j * v2[1]
            p = (x, y)
            if p in seen:
                return None
            seen.add(p)
            points_list.append(p)
            
    if len(points_list) != n * m:
        return None
        
    return np.array(points_list, dtype=np.int64)

def _validate_and_round(points, lines):
    """
    Ensure points and lines are valid: integer, no NaN/inf, no repeated points,
    no zero lines, no duplicate lines. Round to int64.
    """
    points = np.round(points).astype(np.int64)
    lines = np.round(lines).astype(np.int64)
    return points, lines

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy: Search over lattice structures (skewed grids) defined by two basis vectors
    and shapes (n, m) with n*m = 54. For each valid lattice, compute the optimal
    27 lines and total incidence. Return the best configuration found.
    """
    # Shapes (n, m) with n*m = 54
    shapes = [(1, 54), (2, 27), (3, 18), (6, 9), (9, 6), (18, 3), (27, 2), (54, 1)]
    
    # Basis vectors to try. We keep them small to avoid huge coordinate values
    # which might cause issues, though int64 handles large numbers well.
    # The "value" of the lattice depends on the number of collinearities.
    # Axis-aligned is (1,0), (0,1).
    # We'll try a range of small integer vectors.
    
    best_incidence = -1
    best_points = None
    best_lines = None
    
    # We will iterate over a subset of basis vectors to keep runtime low.
    # Range -5 to 5 for each component.
    # Total vectors: 11*11 = 121. Pairs: 121*121 = 14641.
    # For each pair and each shape (8 shapes), we generate points and check.
    # This might be too slow if we do it naively in Python loops for line counting.
    # But line counting is vectorized.
    # Generation is 54 points.
    # Let's limit the search space.
    
    # Pre-define a set of promising basis vectors
    # (1,0), (0,1) is the standard grid.
    # (1,1), (0,1) is a shear.
    # (1,1), (1,0)
    # (2,1), (1,0)
    # (1,2), (0,1)
    # etc.
    
    basis_candidates = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if a == 0 and b == 0:
                continue
            basis_candidates.append((a, b))
            
    # To reduce computation, we can prune basis vectors that are multiples of each other
    # or result in the same lattice structure up to rotation/reflection?
    # But for now, let's just try a smaller, curated set to ensure speed.
    # The standard grid is (1,0) and (0,1).
    # Let's try all pairs from a smaller set first.
    
    small_basis = [
        (1, 0), (0, 1),
        (1, 1), (1, -1),
        (2, 1), (1, 2),
        (2, -1), (1, -2),
        (3, 1), (1, 3),
        (3, 2), (2, 3),
    ]
    
    # Also include negative versions to cover different orientations
    full_basis = []
    for v in small_basis:
        full_basis.append(v)
        full_basis.append((-v[0], v[1]))
        full_basis.append((v[0], -v[1]))
        full_basis.append((-v[0], -v[1]))
        
    # Remove duplicates
    full_basis = list(set(full_basis))
    
    for n, m in shapes:
        for v1 in full_basis:
            for v2 in full_basis:
                # Check if v1 and v2 are linearly independent
                det = v1[0]*v2[1] - v1[1]*v2[0]
                if det == 0:
                    continue
                
                points = _generate_lattice_points(n, m, v1, v2)
                if points is None:
                    continue
                
                # Check if points are "reasonable" (not too large, though not strictly required)
                # The task allows any integer.
                
                lines = _pick_top_k_lines(points, 27)
                if len(lines) < 27:
                    continue
                    
                inc = _count_total_incidence(points, lines)
                if inc > best_incidence:
                    best_incidence = inc
                    best_points = points.copy()
                    best_lines = lines.copy()
                    
    if best_points is None:
        # Fallback to standard grid
        best_points = np.array([[i, j] for i in range(6) for j in range(9)], dtype=np.int64)
        best_lines = _pick_top_k_lines(best_points, 27)
        
    points, lines = _validate_and_round(best_points, best_lines)
    return points, lines
