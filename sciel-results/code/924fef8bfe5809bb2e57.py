import random
import numpy as np
from collections import Counter
from itertools import combinations

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    """
    # Strategy: Use a structured set of points to maximize a single area value.
    # We'll use a grid-based approach with specific rows to create many triangles of equal area.
    
    # Start with a base grid
    base_points = []
    
    # We'll try to build a configuration with many triangles of area 2 (doubled area 4).
    # Using a lattice structure.
    
    # Let's try a rectangular grid with some modifications
    # 6x5 grid = 30 points
    # Let's analyze the area distribution of a 6x5 grid
    
    # Actually, let's try to be smarter. We can use a set of points that creates
    # many collinear triples and a dominant area.
    
    # Approach: Use points on multiple parallel lines to create many triangles
    # with the same base and height.
    
    # Let's try:
    # Row 0: 10 points
    # Row 1: 10 points  
    # Row 2: 10 points
    # This creates triangles with height 1 (doubled area = base * 2)
    
    # Actually, let's use a more efficient structure.
    
    # Let me try a different approach: use a set of points that are vertices
    # of many equal-area triangles.
    
    # Best approach: Use a regular grid and see what area is most frequent.
    
    # 6x5 grid
    grid_points = [(x, y) for x in range(6) for y in range(5)]
    
    # Let's compute the area distribution for this grid
    def compute_area_distribution(points):
        n = len(points)
        areas = Counter()
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # Doubled area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    doubled_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if doubled_area > 0:
                        areas[doubled_area] += 1
        return areas
    
    # Try different grid dimensions
    best_points = grid_points
    best_max_multiplicity = 0
    best_area = 0
    
    # Try various grid configurations
    configs = [
        (6, 5),  # current
        (5, 6),
        (7, 4),  # 28, need 2 more
        (4, 7),
        (10, 3),  # 30
        (3, 10),
        (9, 3),  # 27, need 3 more
        (8, 4),  # 32, too many
        (5, 5),  # 25, need 5 more
    ]
    
    for w, h in configs:
        if w * h != 30:
            continue
            
        points = [(x, y) for x in range(w) for y in range(h)]
        if len(points) != 30:
            continue
            
        areas = compute_area_distribution(points)
        if areas:
            max_mult = max(areas.values())
            if max_mult > best_max_multiplicity:
                best_max_multiplicity = max_mult
                best_points = points
                best_area = [a for a, m in areas.items() if m == max_mult][0]
    
    # Try to improve by perturbing points
    if best_max_multiplicity > 0:
        # Try local search
        for _ in range(100):
            candidate = list(best_points)
            idx = random.randint(0, 29)
            x, y = candidate[idx]
            # Try nearby points
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in candidate:
                            candidate[idx] = (nx, ny)
                            areas = compute_area_distribution(candidate)
                            if areas:
                                max_mult = max(areas.values())
                                if max_mult > best_max_multiplicity:
                                    best_max_multiplicity = max_mult
                                    best_points = candidate
                                    best_area = [a for a, m in areas.items() if m == max_mult][0]
    
    return list(best_points)
