# EVOLVE-BLOCK-START
import numpy as np
from math import gcd


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Strategy:
    Uses a 6x9 grid of points: (x, y) for x in {0..5}, y in {0..8}.
    Enumerates all lines determined by pairs of points, computes incidences,
    then greedily selects 27 lines to maximize total incidences.
    """
    # 1. Generate Points: 6x9 grid
    xs = np.arange(0, 6)
    ys = np.arange(0, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Enumerate all lines through pairs of points
    # For each pair, compute the line (A, B, C) normalized
    line_set = {}  # normalized (A, B, C) -> list of point indices
    
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2):
            # (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = int(y2 - y1)
            B = int(-(x2 - x1))
            C = int((x2 - x1) * y1 - (y2 - y1) * x2)
            
            # Normalize: divide by gcd, make first nonzero positive
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            
            key = (A, B, C)
            if key not in line_set:
                line_set[key] = set()
            line_set[key].add(i)
            line_set[key].add(j)
    
    # Compute incidence count for each line
    line_incidents = {}
    for key, pt_set in line_set.items():
        line_incidents[key] = len(pt_set)
    
    # Sort lines by incidence count (descending)
    sorted_lines = sorted(line_incidents.items(), key=lambda x: -x[1])
    
    # Greedily select 27 lines maximizing total incidences
    # Use a greedy approach: at each step, pick the line with highest marginal gain
    # Since lines can share points, we need to track which points are "covered"
    # But actually, we just want to maximize sum of incidences, and incidences are
    # independent per line (a point on a line counts for that line regardless of other lines)
    # So we just pick the 27 lines with highest incidence counts!
    
    selected = sorted_lines[:27]
    
    lines_array = np.array([k for k, v in selected], dtype=np.int64)
    
    return points, lines_array

# EVOLVE-BLOCK-END
