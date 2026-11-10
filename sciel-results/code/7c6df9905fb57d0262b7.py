"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def _doubled_area(p1, p2, p3):
    """Compute absolute doubled area of triangle with vertices p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

def _count_area_multiplicity(points, target_area):
    """Count how many triples have doubled area equal to target_area."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _doubled_area(points[i], points[j], points[k]) == target_area:
                    count += 1
    return count

def _evaluate(points):
    """Find the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    area_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    area_counts[a] = area_counts.get(a, 0) + 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def construct():
    """
    Construct 30 distinct integer points in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area.
    
    Strategy: Try several candidate configurations based on grid patterns
    and area-targeting heuristics.
    """
    # Candidate area values to target
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    
    best_score = 0
    best_points = None
    
    # Strategy 1: Various grid configurations
    grid_configs = [
        # (width, height) for rectangular grids
        (6, 5), (5, 6), (7, 5), (5, 7), (8, 4), (4, 8),
        (10, 3), (3, 10), (15, 2), (2, 15), (30, 1), (1, 30),
        (4, 8), (8, 4), (5, 6), (6, 5),
    ]
    
    for w, h in grid_configs:
        if w * h < 30:
            continue
        # Generate grid centered at origin
        points = []
        x_start = -w // 2
        y_start = -h // 2
        for x in range(x_start, x_start + w):
            for y in range(y_start, y_start + h):
                if -30 <= x <= 30 and -30 <= y <= 30:
                    points.append((x, y))
                if len(points) == 30:
                    break
            if len(points) == 30:
                break
        if len(points) != 30:
            continue
        # Check distinctness
        if len(set(points)) != 30:
            continue
        score = _evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Strategy 2: Points along a line with small offsets to create many equal areas
    # Try: points at (i, 0) for i in range(15) and (i, 1) for i in range(15)
    # This creates a 15x2 grid
    for offset in range(-10, 11):
        points = []
        for i in range(15):
            x = offset + i
            if -30 <= x <= 30:
                points.append((x, 0))
                points.append((x, 1))
            if len(points) == 30:
                break
        if len(points) != 30 or len(set(points)) != 30:
            continue
        score = _evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points
    
    # Strategy 3: Points on two parallel lines with staggered positions
    for dy in [1, 2, 3]:
        for x_start in range(-15, 0):
            points = []
            for i in range(15):
                x1 = x_start + i
                if -30 <= x1 <= 30:
                    points.append((x1, 0))
            for i in range(15):
                x2 = x_start + i + dy  # staggered
                if -30 <= x2 <= 30:
                    points.append((x2, dy))
            if len(points) != 30 or len(set(points)) != 30:
                continue
            score = _evaluate(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    # Strategy 4: Points forming a triangular lattice pattern
    # Use points where x + 2*y is even (triangular lattice)
    for x_start in range(-20, 5):
        for y_start in range(-15, 5):
            points = []
            for x in range(x_start, x_start + 12):
                for y in range(y_start, y_start + 8):
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        # Triangular lattice: include if (x + 2*y) % 2 == 0
                        if (x + 2 * y) % 2 == 0:
                            points.append((x, y))
                    if len(points) == 30:
                        break
                if len(points) == 30:
                    break
            if len(points) != 30 or len(set(points)) != 30:
                continue
            score = _evaluate(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    # Strategy 5: Dense cluster with controlled area distribution
    # Try points in a small region to get many small-area triangles
    for cx in range(-5, 6):
        for cy in range(-5, 6):
            points = []
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    x, y = cx + dx, cy + dy
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        points.append((x, y))
                    if len(points) == 30:
                        break
                if len(points) == 30:
                    break
            if len(points) != 30 or len(set(points)) != 30:
                continue
            score = _evaluate(points)
            if score > best_score:
                best_score = score
                best_points = points
    
    if best_points is None:
        # Fallback to seed
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
# EVOLVE-BLOCK-END
