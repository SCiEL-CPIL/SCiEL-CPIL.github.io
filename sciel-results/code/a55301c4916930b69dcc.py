# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import combinations


def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    g = gcd(abs(int(A)), gcd(abs(int(B)), abs(int(C))))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (int(A), int(B), int(C))


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    
    Uses a 9x6 grid and greedy line selection to maximize incidences.
    """
    # 1. Generate 54 points: 9x6 grid
    xs = np.arange(0, 9)
    ys = np.arange(0, 6)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    n_points = len(points)
    
    # 2. Generate candidate lines from all pairs of points
    candidate_lines = set()
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
            A = int(y1 - y2)
            B = int(x2 - x1)
            C = int(x1 * y2 - x2 * y1)
            norm = _normalize_line(A, B, C)
            candidate_lines.add(norm)
    
    # Add some additional candidate lines (vertical, horizontal, various slopes)
    # Vertical lines x = c
    for c in range(0, 9):
        candidate_lines.add(_normalize_line(1, 0, -c))
    # Horizontal lines y = c
    for c in range(0, 6):
        candidate_lines.add(_normalize_line(0, 1, -c))
    
    # Add lines with various small slopes
    for m in range(-3, 4):
        for c in range(-5, 15):
            # y = mx + c => mx - y + c = 0
            if m != 0:
                candidate_lines.add(_normalize_line(m, -1, c))
            # Also try other orientations
    for m in range(-3, 4):
        for c in range(-10, 20):
            # x = my + c => x - my - c = 0 (for vertical-ish lines)
            if m != 0:
                candidate_lines.add(_normalize_line(1, -m, -c))
    
    candidate_list = list(candidate_lines)
    
    # 3. Greedy selection of 27 lines maximizing total incidences
    # Compute incidence count for each candidate line
    line_incidents = {}
    for line in candidate_list:
        A, B, C = line
        # Count points on this line
        count = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                count += 1
        line_incidents[line] = count
    
    # Sort lines by incidence count (descending)
    sorted_lines = sorted(candidate_list, key=lambda l: line_incidents[l], reverse=True)
    
    selected_lines = []
    used_points_on_lines = np.zeros(n_points, dtype=np.int64)  # Track how many selected lines pass through each point
    
    # Greedy: pick line that maximizes marginal gain
    # Marginal gain of adding line L = number of points on L (since each point-line pair counts once)
    # But we want to maximize total incidences, which is just sum of points on each selected line.
    # So greedy by total points on line is optimal for this additive objective.
    
    # Actually, since incidence is just sum over lines of (# points on line), 
    # and lines don't interfere with each other in the count, we just pick the 27 lines with highest point counts.
    
    # But we need distinct lines. Let's pick top 27 by point count.
    selected_lines = sorted_lines[:27]
    
    # Verify no duplicates (shouldn't happen since we used a set)
    lines_array = np.array(selected_lines, dtype=np.int64)
    
    # Verify shapes
    assert points.shape == (54, 2), f"Points shape {points.shape}"
    assert lines_array.shape == (27, 3), f"Lines shape {lines_array.shape}"
    
    # Compute incidences to verify
    I = 0
    for pt in points:
        x, y = pt
        for line in lines_array:
            A, B, C = line
            if A * x + B * y + C == 0:
                I += 1
    
    return points, lines_array

# EVOLVE-BLOCK-END
