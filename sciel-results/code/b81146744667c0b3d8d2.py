"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Find all lines (as normalized (slope, intercept) or direction+offset)
    # Use direction vector and a point to uniquely identify a line
    lines = {}  # key: (normalized direction, offset) -> set of point indices
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx == 0 and dy == 0:
                continue
            g = __import__('math').gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Ensure consistent sign: make first non-zero component positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            # Compute a unique identifier for the line
            # Line: point (x1,y1) with direction (dx_n, dy_n)
            # Normal form: dy_n * x - dx_n * y = c
            # c = dy_n * x1 - dx_n * y1
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    # Count lines with exactly 2 points
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    # Try multiple configurations and pick the one with fewest ordinary lines
    candidates = []
    
    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 3x3 grid + 3 extra points on a line
    base = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 more points on a new line, e.g., y=3, x=0,1,2
    extra = [[0, 3], [1, 3], [2, 3]]
    candidates.append(base + extra)
    
    # 4x3 grid with some points moved to create more collinearities
    # Try a parallelogram-like structure
    
    best_points = None
    best_count = float('inf')
    
    for pts in candidates:
        # Check all distinct
        if len(set(tuple(p) for p in pts)) != 12:
            continue
        # Check bounds
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            continue
        # Check not all collinear
        if len(pts) < 3:
            continue
        (x0, y0) = pts[0]
        (x1, y1) = pts[1]
        all_collinear = True
        for i in range(2, len(pts)):
            (x2, y2) = pts[i]
            if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
                all_collinear = False
                break
        if all_collinear:
            continue
        
        c = count_ordinary_lines(pts)
        if c < best_count:
            best_count = c
            best_points = pts
    
    if best_points is None:
        # Fallback to 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
# EVOLVE-BLOCK-END
