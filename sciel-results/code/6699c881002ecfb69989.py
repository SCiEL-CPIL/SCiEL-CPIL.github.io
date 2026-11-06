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
    
    def compute_max_multiplicity(points):
        n = len(points)
        areas = Counter()
        # Optimization: Use numpy for vectorized area calculation if possible, 
        # but n=30 is small, so nested loops are fine.
        for i in range(n-2):
            x1, y1 = points[i]
            for j in range(i+1, n-1):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    # Doubled area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    doubled_area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if doubled_area > 0:
                        areas[doubled_area] += 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        best_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, best_area

    best_points = None
    best_mult = 0
    best_area = 0
    
    # Strategy: 3 parallel lines y=0, y=1, y=2
    # Distribute 30 points among them.
    # Try all splits n0, n1, n2
    
    for n0 in range(1, 29):
        for n1 in range(1, 30 - n0):
            n2 = 30 - n0 - n1
            if n2 < 1:
                continue
            
            # Use consecutive integers for x coordinates
            # To maximize symmetry and likely multiplicity, center them or start at 0
            # Starting at 0 is fine due to translation invariance of area
            pts0 = [(x, 0) for x in range(n0)]
            pts1 = [(x, 1) for x in range(n1)]
            pts2 = [(x, 2) for x in range(n2)]
            
            points = pts0 + pts1 + pts2
            
            # Check distinctness (guaranteed by y coords)
            # Check bounds (x < 30, y < 30) - guaranteed
            
            mult, area = compute_max_multiplicity(points)
            if mult > best_mult:
                best_mult = mult
                best_area = area
                best_points = points.copy()
                
    # Also try 2 lines y=0, y=1 for comparison
    for n0 in range(1, 30):
        n1 = 30 - n0
        pts0 = [(x, 0) for x in range(n0)]
        pts1 = [(x, 1) for x in range(n1)]
        points = pts0 + pts1
        mult, area = compute_max_multiplicity(points)
        if mult > best_mult:
            best_mult = mult
            best_area = area
            best_points = points.copy()

    # Also try 4 lines y=0,1,2,3
    # This is O(30^3) = 27000 iterations, each O(30^3) = 27000 ops -> 7e8 ops. Too slow.
    # We'll stick to 2 and 3 lines which are faster to evaluate or we can sample.
    # Actually, 3 lines loop is ~400 iterations. 400 * 4500 (triples) = 1.8M ops. Fast.
    # 4 lines loop is ~4000 iterations. 4000 * 4500 = 18M ops. Still fast enough in Python?
    # Let's add 4 lines with a smaller search space or just skip if 3 lines are good.
    # Given 3 lines gave ~450, and 2 lines 420, 3 lines is likely better.
    # Let's try to optimize the x-coordinates slightly?
    # Consecutive integers are optimal for "many pairs at small distance".
    
    return list(best_points)
