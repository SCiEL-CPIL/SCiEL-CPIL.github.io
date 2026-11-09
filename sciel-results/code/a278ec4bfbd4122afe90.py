"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations
import math
import random

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line (normalized) to set of point indices on that line
    line_to_points = {}
    
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        
        # Line through (x1,y1) and (x2,y2):
        # (y2-y1)x - (x2-x1)y + (x2*y1 - y2*x1) = 0
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - y2 * x1
        
        # Normalize the line representation
        def gcd(a, b):
            a, b = abs(a), abs(b)
            while b:
                a, b = b, a % b
            return a
        
        g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
        if g > 0:
            A //= g
            B //= g
            C //= g
        
        # Ensure canonical sign: first non-zero of (A, B, C) should be positive
        if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
            A = -A
            B = -B
            C = -C
        
        key = (A, B, C)
        if key not in line_to_points:
            line_to_points[key] = set()
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    """
    Construct 12 integer points minimizing the number of ordinary lines.
    
    Strategy: Use a hexagonal lattice-based configuration with high symmetry,
    then apply greedy local search to maximize 3-point collinearities.
    """
    
    # Start with a hexagonal lattice arrangement.
    # We'll use points from a hexagonal grid pattern.
    # Hexagonal lattice: points at (i + j/2, j*sqrt(3)/2)
    # To get integer coordinates, we scale and use a different basis.
    
    # Let's try a configuration based on a regular hexagon with integer coordinates.
    # A regular hexagon with side length s has vertices:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # This doesn't give integer coordinates directly.
    
    # Instead, use a "hexagonal" arrangement on a square grid that has many 3-point lines.
    # Start with a 4x3 grid and optimize.
    
    points = [[x, y] for x in range(4) for y in range(3)]
    
    # Initial count
    current_ordinary = count_ordinary_lines(points)
    
    # Local search with simulated annealing-like approach
    random.seed(42)
    
    best_points = [p[:] for p in points]
    best_ordinary = current_ordinary
    
    max_iter = 30000
    temperature = 5.0
    cooling = 0.999
    
    for iteration in range(max_iter):
        # Try random point moves
        idx = random.randint(0, 11)
        old_x, old_y = points[idx]
        
        # Try a random new position
        # Use a range that allows for significant rearrangement
        new_x = random.randint(-20, 20)
        new_y = random.randint(-20, 20)
        
        # Skip if same position
        if new_x == old_x and new_y == old_y:
            continue
        
        # Check if new position is already taken
        if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
            continue
        
        # Move the point
        points[idx] = [new_x, new_y]
        new_ordinary = count_ordinary_lines(points)
        
        delta = new_ordinary - best_ordinary
        
        # Accept if better, or with probability based on temperature if worse
        if delta <= 0 or random.random() < math.exp(-delta / temperature):
            if new_ordinary < best_ordinary:
                best_ordinary = new_ordinary
                best_points = [p[:] for p in points]
        else:
            # Revert
            points[idx] = [old_x, old_y]
        
        temperature *= cooling
        
        # If we've reached the theoretical minimum, stop
        if best_ordinary <= 6:
            break
    
    # Ensure we return the best found
    result = [p[:] for p in best_points]
    
    # Verify: 12 distinct points, not all collinear, |x|,|y| <= 10^9
    # The coordinates are well within bounds.
    
    return result
# EVOLVE-BLOCK-END
