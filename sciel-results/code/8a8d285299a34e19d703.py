import numpy as np
from math import gcd
import time


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences on a 9x6 integer grid.
    54 points, select 27 lines to maximize total incidences.
    Strategy: enumerate all lines through grid point pairs, count incidences,
    select top 27 by incidence count.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    # Create 9x6 grid
    xs = np.arange(1, 10)  # 1 to 9
    ys = np.arange(1, 7)   # 1 to 6
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Find all unique lines through pairs of points
    # A line through (x1,y1) and (x2,y2) has direction (dx, dy) = (x2-x1, y2-y1)
    # Canonical form: A*x + B*y + C = 0 where (A,B) is perpendicular to direction
    # A = dy, B = -dx, C = -(A*x1 + B*y1)
    # Normalize: divide by gcd(|A|,|B|,|C|), make first nonzero positive
    
    lines_dict = {}  # canonical (A,B,C) -> set of point indices on line
    
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            # Normalize
            g = gcd(abs(A), abs(B))
            g = gcd(g, abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            # Make first nonzero entry positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines_dict:
                lines_dict[key] = set()
            lines_dict[key].add(i)
            lines_dict[key].add(j)
    
    # Now for each line, count all points on it
    line_incidences = []
    for (A, B, C), _ in lines_dict.items():
        # Count all points on this line
        count = 0
        for idx in range(n):
            x, y = points[idx]
            if A * x + B * y + C == 0:
                count += 1
        line_incidences.append((count, (A, B, C)))
    
    # Sort by incidence count descending
    line_incidences.sort(key=lambda x: -x[0])
    
    # Select top 27 lines
    selected_lines = []
    for count, (A, B, C) in line_incidences[:27]:
        selected_lines.append((A, B, C))
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines
