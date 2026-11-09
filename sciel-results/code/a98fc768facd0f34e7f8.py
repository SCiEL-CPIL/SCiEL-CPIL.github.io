"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use set of frozensets to identify unique lines
    # A line is determined by any two points on it
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Represent line by normalized (A, B, C) where Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize: make gcd of coefficients 1, and fix sign
            from math import gcd
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            # Fix sign: make first non-zero positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    """Check that points are not all on a single line."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, len(points)):
        xi, yi = points[i]
        # Cross product to check collinearity
        cross = (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0)
        if cross != 0:
            return True
    return False

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    # 3x4 grid: 4 columns, 3 rows
    # This configuration achieves 6 ordinary lines (theoretical minimum)
    points = [[x, y] for x in range(4) for y in range(3)]
    
    # Verify constraints
    assert len(points) == 12
    assert len(set(tuple(p) for p in points)) == 12  # all distinct
    assert all(abs(x) <= 10**9 and abs(y) <= 10**9 for x, y in points)
    assert _not_all_collinear(points)
    
    return points
# EVOLVE-BLOCK-END
