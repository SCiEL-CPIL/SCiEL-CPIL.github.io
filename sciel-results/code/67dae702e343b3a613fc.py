"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # doubled area = |det|
                a = pts[i]
                b = pts[j]
                c = pts[k]
                d = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    # Count multiplicity of each area value
    from collections import Counter
    cnt = Counter(areas)
    return max(cnt.values())

def _generate_grid_config(num_rows, num_cols, y_start=0, x_start=0):
    """Generate points on a grid: num_rows parallel lines, num_cols points each."""
    points = []
    for y in range(num_rows):
        for x in range(num_cols):
            points.append((x_start + x, y_start + y))
    return points

def _evaluate_grid(num_rows, num_cols, y_start=0, x_start=0):
    """Evaluate a grid configuration and return its score."""
    points = _generate_grid_config(num_rows, num_cols, y_start, x_start)
    if len(points) != 30:
        return 0
    return _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    # Start with the known good baseline: 3x10 grid
    best_points = _generate_grid_config(3, 10)
    best_score = _score(best_points)
    
    # Try various grid configurations that give exactly 30 points
    # 30 = 1*30, 2*15, 3*10, 5*6, 6*5, 10*3, 15*2, 30*1, 4*7.5(no), 7*4.28(no)
    # We can only use integer rows/cols.
    grid_configs = [
        (3, 10),  # baseline
        (5, 6),
        (6, 5),
        (2, 15),
        (15, 2),
        (1, 30),
        (30, 1),
        (10, 3),
        (4, 7),   # 28 points, need 2 more? No, 4*7=28.
        (7, 4),   # 28 points.
    ]
    
    # 4x7 is 28 points. We need 30.
    # So we can't use 4x7 directly as a full grid.
    # But we can use a 4x7 grid + 2 points?
    # Or just stick to factors of 30.
    
    # Let's add a custom generator for non-rectangular or sparse grids if needed,
    # but for now, let's evaluate the factors of 30.
    
    # Add 4x7 + 2 points?
    # Let's try to append 2 points to a 4x7 grid to see if it helps.
    # But the symmetry is broken.
    
    # Let's just evaluate the factor grids first.
    for rows, cols in grid_configs:
        if rows * cols != 30:
            continue
        if time.time() > deadline:
            break
        s = _evaluate_grid(rows, cols)
        if s > best_score:
            best_score = s
            best_points = _generate_grid_config(rows, cols)
    
    # Try some shifted/offset grids to see if they help
    # For 3x10, try different spacings
    if time.time() < deadline - 10:
        for spacing in [1, 2, 3]:
            if time.time() > deadline:
                break
            points = []
            for y in range(3):
                for x in range(10):
                    points.append((x * spacing, y * spacing))
            if len(set(points)) == 30:
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points
    
    # Try a few more structured patterns
    # 5x6 grid with different spacing
    if time.time() < deadline - 5:
        for spacing in [1, 2]:
            if time.time() > deadline:
                break
            points = []
            for y in range(5):
                for x in range(6):
                    points.append((x * spacing, y * spacing))
            if len(set(points)) == 30:
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points
                    
    # Try 2x15 with spacing
    if time.time() < deadline - 5:
        for spacing in [1, 2]:
            if time.time() > deadline:
                break
            points = []
            for y in range(2):
                for x in range(15):
                    points.append((x * spacing, y * spacing))
            if len(set(points)) == 30:
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points

    return best_points
# EVOLVE-BLOCK-END
