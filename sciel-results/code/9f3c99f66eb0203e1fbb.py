"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 points."""
    n = len(points)
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            # Normalize the direction vector
            from math import gcd
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            key = (dx, dy, points[i][0] * dy - points[i][1] * dx)
            line_counts[key] += 1
    
    # Each line with k points is counted C(k,2) times
    count = 0
    for k in line_counts.values():
        # Find how many points are on this line
        # k = C(m, 2) where m is the number of points
        # Solve m(m-1)/2 = k for m
        # m^2 - m - 2k = 0
        # m = (1 + sqrt(1 + 8k)) / 2
        import math
        disc = 1 + 8 * k
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc:
            m = (1 + sqrt_disc) // 2
            if m == 4:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # AG(2,4) has 16 points and 20 lines of size 4
    # We can embed it using a finite field construction
    # Points are (a, b) where a, b in GF(4)
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    
    # Map GF(4) to integers: 0->0, 1->1, alpha->2, alpha+1->3
    # But we need to preserve collinearity, so use a different approach
    
    # Use the standard AG(2,4) structure:
    # Points: (i, j) for i, j in {0, 1, 2, 3}
    # Lines: for each slope m in {0, 1, 2, 3, inf} and intercept b in {0, 1, 2, 3}
    # Line with slope m: y = m*x + b (mod 4)
    # Line with slope inf: x = b
    
    # This gives 5*4 = 20 lines of size 4
    
    # Now map to integer coordinates while preserving collinearity
    # Use a linear transformation: (x, y) -> (a*x + b*y, c*x + d*y)
    # We need to choose a, b, c, d such that no 5 points are collinear
    
    # Try a simple mapping first: just use (x, y) as is
    points_grid = [[i, j] for i in range(4) for j in range(4)]
    
    # Scale to use more of the coordinate range
    scale = 250
    points_scaled = [[i * scale, j * scale] for i in range(4) for j in range(4)]
    
    # Verify the count
    count = _count_lines_exactly_4(points_scaled)
    
    # If we get 20, we're done
    if count >= 20:
        return points_scaled
    
    # Try other linear transformations
    # (x, y) -> (x + y, x - y)
    points_diag = [[(i + j) * scale, (i - j) * scale] for i in range(4) for j in range(4)]
    count = _count_lines_exactly_4(points_diag)
    if count >= 20:
        return points_diag
    
    # (x, y) -> (2*x + y, x + 2*y)
    points_mixed = [[(2*i + j) * scale, (i + 2*j) * scale] for i in range(4) for j in range(4)]
    count = _count_lines_exactly_4(points_mixed)
    if count >= 20:
        return points_mixed
    
    # Return the best we found
    return points_grid
# EVOLVE-BLOCK-END
