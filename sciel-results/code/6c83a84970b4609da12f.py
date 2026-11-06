"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    # Map each line (through at least 2 points) to the set of points on it
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            # Normalize: ensure consistent sign
            if dx_n < 0:
                dx_n = -dx_n
                dy_n = -dy_n
            elif dx_n == 0 and dy_n < 0:
                dy_n = -dy_n
            # Line: y - y1 = (dy_n/dx_n)(x - x1) if dx_n != 0
            # Use: dy_n * x - dx_n * y = dy_n * x1 - dx_n * y1
            if dx_n != 0:
                c = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, c)
            else:
                # Vertical line: x = x1
                key = (0, 1, x1)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Try several candidate configurations
    candidates = []
    
    # 1. 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid (same set, different order - same result)
    # Skip, same as above
    
    # 3. 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 4. 6x2 grid
    candidates.append([[x, y] for x in range(2) for y in range(6)])
    
    # 5. Hexagonal-like: points in two rows with offset
    # Row 0: x = 0,1,2,3,4,5
    # Row 1: x = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 -> use 2x to make integer
    # Scale by 2:
    pts_hex = []
    for x in range(6):
        pts_hex.append([2*x, 0])
    for x in range(6):
        pts_hex.append([2*x+1, 2])
    candidates.append(pts_hex)
    
    # 6. Points on a conic: y = x^2 mod something? No, use actual parabola
    # y = x^2 for x in some range - but these are all on one line? No, parabola.
    # Actually all on y=x^2, which is a curve, not a line. Lines through pairs.
    # For 12 points on a parabola, every line through two points intersects the parabola in at most 2 points (unless tangent), so all lines are ordinary! That's 66 lines. Bad.
    
    # 7. Use a known configuration: 4x3 grid is same as 3x4.
    # Try: 3 points on each of 4 parallel lines, with offsets
    # Lines at y=0,1,2,3
    # y=0: x=0,1,2
    # y=1: x=0,1,2  
    # y=2: x=0,1,2
    # y=3: x=0,1,2
    # This is 4x3 grid, same as #1.
    
    # 8. Try a configuration with more collinear points
    # 3 points on y=0, 3 on y=1, 3 on y=2, 3 on y=3, but with different x offsets
    # to create more diagonal lines with 3+ points
    # y=0: x=0,1,2
    # y=1: x=1,2,3
    # y=2: x=2,3,4
    # y=3: x=3,4,5
    # This creates diagonal lines with 3 points: (0,0),(1,1),(2,2),(3,3) - 4 points!
    # (1,0),(2,1),(3,2),(4,3) - 4 points
    # (2,0),(3,1),(4,2),(5,3) - 4 points
    # etc.
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x+y, y])
    candidates.append(pts_diag)
    
    # 9. Try: 6 points on y=0 at x=0..5, 6 points on y=2 at x=0..5
    # but this is similar to 2x6.
    
    # 10. Try a configuration based on the Sylvester-Gallai theorem extremal examples
    # For 12 points, try to maximize 3-point lines.
    # Each 3-point line reduces the count by 1 compared to all-pairs being ordinary.
    # Total pairs = 66. If we have L3 3-point lines, L4 4-point lines, etc.
    # Ordinary lines = lines with exactly 2 points.
    
    # Let me try: 3x4 grid but with different spacing to create more diagonals
    # Standard 3x4: x in {0,1,2,3}, y in {0,1,2}
    # Diagonals with slope 1: (0,0)-(1,1)-(2,2) [3 pts], (1,0)-(2,1)-(3,2) [3 pts], (0,1)-(1,2) [2 pts], (0,2)-(1,1) [2 pts], (1,2)-(2,1) [2 pts], (2,2)-(3,1) [2 pts], (3,2)-(2,1) same, etc.
    
    # 11. Try a 12-point configuration from known results
    # The "near-pencil" is bad. 
    # Try: 4 points on each of 3 parallel lines, with careful placement
    # y=0: x=0,1,2,3
    # y=1: x=0,1,2,3
    # y=2: x=0,1,2,3
    # This is 3x4 grid, already tried.
    
    # 12. Try: points at vertices and midpoints of a regular hexagon (integer approx)
    # Regular hexagon with "radius" R:
    # (R,0), (R/2, R*sqrt(3)/2), (-R/2, R*sqrt(3)/2), (-R,0), (-R/2, -R*sqrt(3)/2), (R/2, -R*sqrt(3)/2)
    # Use R=10^6, sqrt(3)/2 ≈ 0.866, R/2 = 5*10^5
    R = 10**6
    hex_pts = [
        [R, 0],
        [R//2, int(R*0.866025)],
        [-R//2, int(R*0.866025)],
        [-R, 0],
        [-R//2, -int(R*0.866025)],
        [R//2, -int(R*0.866025)],
    ]
    # Add 6 more points: center and 6 more? 
    # Or add 6 points at double the radius in the same directions
    hex_pts2 = list(hex_pts)
    for p in hex_pts:
        hex_pts2.append([2*p[0], 2*p[1]])
    # Check bounds: 2*R = 2*10^6, 2*int(R*0.866) ≈ 1.73*10^6, all < 10^9. OK.
    candidates.append(hex_pts2)
    
    # 13. Try: 3 points on each of 4 lines through a common point (near-pencil)
    # All 12 points on 4 lines through origin, 3 per line.
    # But then many lines through pairs of points on different lines will be ordinary.
    # Lines through origin: 4 lines, each with 3 points (not ordinary)
    # Lines through one point from line i and one from line j: these are ordinary unless they pass through a third point.
    # This likely gives many ordinary lines.
    
    # Evaluate all candidates
    best_pts = None
    best_count = float('inf')
    
    for idx, pts in enumerate(candidates):
        if time.time() > deadline:
            break
        # Verify: 12 distinct points, |x|,|y| <= 10^9, not all collinear
        if len(pts) != 12:
            continue
        if len(set(tuple(p) for p in pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in pts):
            continue
        # Check not all collinear
        if len(pts) >= 3:
            (x0,y0),(x1,y1),(x2,y2) = pts[0], pts[1], pts[2]
            if (x1-x0)*(y2-y0) == (y1-y0)*(x2-x0):
                # Check if all collinear
                all_col = True
                for i in range(3, 12):
                    xi, yi = pts[i]
                    if (x1-x0)*(yi-y0) != (y1-y0)*(xi-x0):
                        all_col = False
                        break
                if all_col:
                    continue
        count = count_ordinary_lines(pts)
        if count < best_count:
            best_count = count
            best_pts = pts
    
    if best_pts is None:
        # Fallback: return 3x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    
    return best_pts
# EVOLVE-BLOCK-END
