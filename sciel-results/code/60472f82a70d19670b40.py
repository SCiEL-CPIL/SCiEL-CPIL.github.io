"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line (normalized) to set of point indices on that line
    # A line is represented by (A, B, C) where Ax + By + C = 0, normalized
    # For integer points, we can represent a line by the normalized (A, B, C)
    
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
        # Find gcd of A, B, C
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
    
    Strategy: Start with a regular hexagon-like configuration on a hexagonal lattice,
    then use local search to swap/move points to maximize 3-point collinearities.
    
    The theoretical minimum for 12 points over reals is 6 ordinary lines.
    We aim to achieve 6 or close to it with integer coordinates.
    
    A known configuration achieving 6 ordinary lines for 12 points uses the
    vertices of a regular hexagon plus its center and other symmetric points,
    but we need integer coordinates.
    
    Let's try a configuration based on the hexagonal lattice.
    """
    
    # Start with a hexagonal lattice arrangement.
    # Regular hexagon vertices at distance r from center:
    # (r, 0), (r/2, r*sqrt(3)/2), ... but we need integers.
    
    # Instead, use a known good configuration for 12 points with few ordinary lines.
    # One approach: use points from a 3x4 grid but rearrange.
    
    # Let's try a configuration that has many 3-point lines.
    # The Hesse configuration has 9 points with 12 lines, each with 3 points.
    # For 12 points, we can extend this.
    
    # Start with a base configuration and optimize via local search.
    
    # Base: use points on a hexagonal lattice pattern.
    # Hexagonal lattice points: (i + j/2, j*sqrt(3)/2) for integers i, j.
    # To get integer coords, scale: use (2i + j, j*sqrt(3)) - not integer.
    
    # Alternative: use a rectangular grid with some perturbations.
    
    # Let's start with a 4x3 grid and try to improve.
    points = [[x, y] for x in range(4) for y in range(3)]
    
    # Initial count
    current_ordinary = count_ordinary_lines(points)
    
    # Local search: try moving each point to a nearby integer position
    # and accept if it reduces the number of ordinary lines.
    
    import random
    random.seed(42)
    
    max_iter = 50000
    best_points = [p[:] for p in points]
    best_ordinary = current_ordinary
    
    for iteration in range(max_iter):
        improved = False
        
        # Try random point moves
        for _ in range(10):
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            
            # Try a random new position in a reasonable range
            # Keep coordinates bounded
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            
            # Skip if same position
            if new_x == old_x and new_y == old_y:
                continue
            
            # Check if new position is already taken
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
                continue
            
            # Move the point
            points[idx] = [new_x, new_y]
            new_ordinary = count_ordinary_lines(points)
            
            if new_ordinary < best_ordinary:
                best_ordinary = new_ordinary
                best_points = [p[:] for p in points]
                improved = True
            else:
                # Revert
                points[idx] = [old_x, old_y]
        
        if not improved and iteration > 100:
            # Reset to best and try a different strategy
            points = [p[:] for p in best_points]
            # Occasionally try a bigger jump
            if iteration % 50 == 0:
                idx = random.randint(0, 11)
                old_x, old_y = points[idx]
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                if not any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
                    points[idx] = [new_x, new_y]
                    new_ordinary = count_ordinary_lines(points)
                    if new_ordinary < best_ordinary:
                        best_ordinary = new_ordinary
                        best_points = [p[:] for p in points]
                    else:
                        points[idx] = [old_x, old_y]
        
        if best_ordinary <= 6:
            break
    
    # Ensure we return the best found
    result = [p[:] for p in best_points]
    
    # Verify: 12 distinct points, not all collinear, |x|,|y| <= 10^9
    # The coordinates are well within bounds.
    
    return result
# EVOLVE-BLOCK-END
