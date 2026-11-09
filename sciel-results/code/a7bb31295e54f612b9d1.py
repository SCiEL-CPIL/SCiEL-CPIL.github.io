"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random
import math

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - y2 * x1
        
        # Normalize using gcd
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
        
        # Canonical sign
        if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
            A = -A
            B = -B
            C = -C
        
        key = (A, B, C)
        if key not in line_to_points:
            line_to_points[key] = set()
        line_to_points[key].add(i)
        line_to_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def generate_hexagonal_config(s=1):
    """Generate a hexagonally-symmetric configuration of 12 points."""
    # Use a hexagonal lattice pattern scaled by s
    # Points: center, 6 at distance s, 6 at distance 2s in alternating directions
    # For integer coords, use a skewed coordinate system
    
    # Hexagonal lattice basis: e1 = (2, 0), e2 = (1, 2) (scaled for integers)
    # Actually, let's use a simpler approach with known good configurations
    
    # Configuration 1: 3x4 grid with some rearrangement
    # Configuration 2: Hexagon-like with integer coords
    
    # Let's use points that form a "hexagonal" pattern:
    # Layer 0: (0, 0)
    # Layer 1: 6 points around center
    # Layer 2: 5 more points
    
    # For integer coordinates, use:
    # Center: (0, 0)
    # Ring 1: (s, 0), (0, s), (-s, 0), (0, -s), (s, s), (-s, -s) - not hexagonal
    # Better: use a diamond/hexagonal shape
    
    # Known good config: use points from {0,1,2,3} x {0,1,2} but permute
    # Or use a configuration with many 3-point lines
    
    # Let's try: 4 points on x-axis, 4 on y-axis, 4 on diagonal
    # This creates many 3-point lines
    
    # Config A: Cross pattern
    # (-3,0), (-2,0), (-1,0), (0,0), (1,0), (2,0), (3,0) - 7 points on x-axis
    # Too many collinear
    
    # Config B: 3 parallel lines with 4 points each
    # Line y=0: x = 0,1,2,3
    # Line y=1: x = 0,1,2,3  
    # Line y=2: x = 0,1,2,3
    # This is just a 4x3 grid
    
    # Config C: Use a "star" pattern with center and 11 points around
    
    # Let me try a specific known configuration:
    # Points: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (3,0), (3,1), (3,2)
    # This is a 4x3 grid
    
    # Better: use a configuration with more 3-point diagonals
    # Try: 
    # Row 0: (0,0), (1,0), (2,0), (3,0)
    # Row 1: (0,1), (1,1), (2,1), (3,1)
    # Row 2: (0,2), (1,2), (2,2), (3,2)
    # Diagonals: (0,0)-(1,1)-(2,2), (1,0)-(2,1)-(3,2), (0,1)-(1,2), etc.
    
    # Let's generate multiple candidate configs and pick the best
    
    candidates = []
    
    # Candidate 1: 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # Candidate 2: 3x4 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Candidate 3: Hexagon-like with center
    # Center + 6 points at hex positions + 5 more
    # Use integer approximations of hex vertices
    hex_pts = [
        [0, 0],       # center
        [2, 0], [1, 2], [-1, 2], [-2, 0], [-1, -2], [1, -2],  # ring 1
        [4, 0], [2, 4], [-2, 4], [-4, 0], [-2, -4], [2, -4]   # ring 2 (only need 5 more)
    ]
    # Take first 12
    candidates.append(hex_pts[:12])
    
    # Candidate 4: Symmetric cross
    cross_pts = [
        [0, 0],
        [1, 0], [-1, 0], [2, 0], [-2, 0],
        [0, 1], [0, -1], [0, 2], [0, -2],
        [1, 1], [-1, -1], [2, 1]
    ]
    candidates.append(cross_pts)
    
    # Candidate 5: Two parallel lines with offsets
    # Line y=0: 6 points, Line y=1: 6 points
    # But arrange to create diagonals
    cand5 = []
    for x in range(6):
        cand5.append([x, 0])
    for x in range(3, 9):
        cand5.append([x, 1])
    candidates.append(cand5)
    
    # Candidate 6: Triangular arrangement
    # Row 0: 4 points, Row 1: 4 points, Row 2: 4 points (shifted)
    cand6 = []
    for i in range(4):
        cand6.append([i, 0])
    for i in range(4):
        cand6.append([i+1, 1])
    for i in range(4):
        cand6.append([i+2, 2])
    candidates.append(cand6)
    
    return candidates

def construct():
    """
    Construct 12 integer points minimizing the number of ordinary lines.
    
    Strategy: Generate multiple candidate configurations based on hexagonal symmetry,
    evaluate them, then use local search to improve the best one.
    """
    
    # Generate candidate configurations
    candidates = generate_hexagonal_config()
    
    # Evaluate each candidate
    best_points = None
    best_ordinary = float('inf')
    
    for cand in candidates:
        # Check validity: 12 distinct points, not all collinear
        if len(cand) != 12:
            continue
        if len(set(tuple(p) for p in cand)) != 12:
            continue
        # Check not all collinear
        collinear = True
        for i in range(2, 12):
            (x0, y0), (x1, y1), (x2, y2) = cand[0], cand[1], cand[i]
            if (x1-x0)*(y2-y0) != (y1-y0)*(x2-x0):
                collinear = False
                break
        if collinear:
            continue
        
        ordinary = count_ordinary_lines(cand)
        if ordinary < best_ordinary:
            best_ordinary = ordinary
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        # Fallback: 4x3 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_ordinary = count_ordinary_lines(best_points)
    
    # Local search to improve
    random.seed(42)
    points = [p[:] for p in best_points]
    
    max_iter = 20000
    for iteration in range(max_iter):
        improved = False
        
        # Try swapping two points with nearby random positions
        for _ in range(20):
            idx = random.randint(0, 11)
            old_x, old_y = points[idx]
            
            # Try random new position
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            
            if new_x == old_x and new_y == old_y:
                continue
            
            if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(points) if i != idx):
                continue
            
            points[idx] = [new_x, new_y]
            new_ordinary = count_ordinary_lines(points)
            
            if new_ordinary < best_ordinary:
                best_ordinary = new_ordinary
                best_points = [p[:] for p in points]
                improved = True
            else:
                points[idx] = [old_x, old_y]
        
        if not improved and iteration > 50:
            points = [p[:] for p in best_points]
        
        if best_ordinary <= 6:
            break
    
    result = [p[:] for p in best_points]
    return result
# EVOLVE-BLOCK-END
