"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from collections import Counter

def _score(points):
    """Compute the max multiplicity of a positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        det = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if det > 0:
            areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _grid_points(rows, cols, dx=1, dy=1, offset_x=0, offset_y=0):
    """Generate a rectangular grid of points."""
    points = []
    for i in range(rows):
        for j in range(cols):
            x = offset_x + j * dx
            y = offset_y + i * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                points.append((x, y))
    return points

def construct():
    best_points = None
    best_score = 0
    
    # Try various grid configurations
    for rows in range(3, 11):
        for cols in range(3, 11):
            if rows * cols != 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    # Check bounds
                    width = (cols - 1) * dx
                    height = (rows - 1) * dy
                    if width > 60 or height > 60:
                        continue
                    # Try different offsets to center the grid
                    for ox in range(-5, 6):
                        for oy in range(-5, 6):
                            points = _grid_points(rows, cols, dx, dy, ox, oy)
                            if len(points) != 30:
                                continue
                            # Check all points are in bounds and distinct
                            if len(set(points)) != 30:
                                continue
                            if any(x < -30 or x > 30 or y < -30 or y > 30 for x, y in points):
                                continue
                            s = _score(points)
                            if s > best_score:
                                best_score = s
                                best_points = points[:]
    
    # Also try some structured non-grid patterns
    # Diamond/hexagonal-like patterns
    for scale in [1, 2, 3]:
        for ox in range(-10, 11, 2):
            for oy in range(-10, 11, 2):
                points = []
                # Hexagonal packing approximation
                for i in range(6):
                    for j in range(5):
                        x = ox + j * scale + (i % 2) * (scale // 2)
                        y = oy + i * scale
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            points.append((x, y))
                if len(points) == 30 and len(set(points)) == 30:
                    s = _score(points)
                    if s > best_score:
                        best_score = s
                        best_points = points[:]
    
    if best_points is None:
        # Fallback to seed
        return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
