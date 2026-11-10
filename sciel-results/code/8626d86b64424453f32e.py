import numpy as np
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """
    Normalize the line coefficients (A, B, C) such that:
    1. gcd(|A|, |B|, |C|) == 1
    2. The first non-zero coefficient is positive.
    Returns a tuple (A, B, C).
    """
    if A == 0 and B == 0:
        return (0, 0, 1) # Invalid line, but handle gracefully
        
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Make first non-zero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
        
    return (A, B, C)

def _line_from_points(p1, p2):
    """
    Calculate line coefficients (A, B, C) for line passing through p1 and p2.
    p1 = (x1, y1), p2 = (x2, y2)
    Equation: (y2 - y1)x - (x2 - x1)y + (x2*y1 - x1*y2) = 0
    So A = y2 - y1, B = -(x2 - x1), C = x2*y1 - x1*y2
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = -(x2 - x1)
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)

def _count_incidence(line, points):
    """
    Count how many points lie on the given line.
    line: (A, B, C)
    points: Nx2 array
    """
    A, B, C = line
    # Check A*x + B*y + C == 0
    # Vectorized operation
    vals = points[:, 0] * A + points[:, 1] * B + C
    return np.sum(vals == 0)

def szemeredi_trotter_incidences():
    """
    Returns a set of 54 points and 27 lines maximizing incidences.
    Uses a grid of points and selects the top 27 lines by incidence count.
    """
    # Define a 6x9 grid
    # Points: (x, y) for x in [0, 5], y in [0, 8]
    # Total points: 6 * 9 = 54
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate all unique lines defined by pairs of points
    # There are 54 points, so ~1400 pairs. This is fast.
    candidate_lines = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            line = _line_from_points(tuple(points[i]), tuple(points[j]))
            candidate_lines.add(line)
    
    # Calculate incidences for each candidate line
    line_scores = []
    for line in candidate_lines:
        score = _count_incidence(line, points)
        line_scores.append((score, line))
        
    # Sort by score descending
    line_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 27 unique lines
    selected_lines = []
    seen_lines = set()
    for score, line in line_scores:
        if line not in seen_lines:
            selected_lines.append(line)
            seen_lines.add(line)
            if len(selected_lines) == 27:
                break
                
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines
