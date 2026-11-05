"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import gcd

def count_two_point_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    # Map each line to the set of point indices on it
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    two_point_lines = sum(1 for pts in line_to_points.values() if len(pts) == 2)
    return two_point_lines

def all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return False
    return True

def construct():
    """
    Construct 12 points minimizing the number of 2-point lines.
    
    We search over several candidate configurations inspired by finite geometry
    and grid structures, then return the one with the fewest 2-point lines.
    """
    
    candidates = []
    
    # 1. 4x3 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 3x4 grid (same as above, transposed)
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. AG(2,3) base + 3 extra points in various positions
    ag23 = [[i, j] for i in range(3) for j in range(3)]
    
    # Add column x=3
    candidates.append(ag23 + [[3, j] for j in range(3)])
    
    # Add row y=3
    candidates.append(ag23 + [[i, 3] for i in range(3)])
    
    # Add diagonal points
    candidates.append(ag23 + [[3, 0], [3, 1], [3, 2]])
    candidates.append(ag23 + [[0, 3], [1, 3], [2, 3]])
    candidates.append(ag23 + [[3, 0], [0, 3], [3, 3]])
    candidates.append(ag23 + [[3, 0], [3, 3], [0, 3]])
    candidates.append(ag23 + [[3, 1], [1, 3], [3, 3]])
    candidates.append(ag23 + [[2, 3], [3, 2], [3, 3]])
    
    # 4. Try a 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])
    
    # 5. Try a configuration with more 3-point lines
    # Use points: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2) - already 4x3
    
    # 6. Try: 9 points in 3x3 + 3 points forming another triangle
    candidates.append(ag23 + [[4, 0], [4, 1], [4, 2]])
    candidates.append(ag23 + [[0, 4], [1, 4], [2, 4]])
    candidates.append(ag23 + [[4, 0], [0, 4], [4, 4]])
    
    # 7. Try symmetric cross-like
    base = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
            [0, 1], [5, 1],
            [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2]]
    # That's 15, too many. Let me be more careful.
    
    # 8. Try: two parallel lines of 4, and two more points
    # Line y=0: (0,0),(1,0),(2,0),(3,0)
    # Line y=1: (0,1),(1,1),(2,1),(3,1)
    # Line y=2: (0,2),(1,2),(2,2),(3,2) -> that's 4x3 again
    
    # 9. Try: 3 points on y=0, 3 on y=1, 3 on y=2, 3 on x=3
    # But (3,0),(3,1),(3,2) would be shared
    # So: (0,0),(1,0),(2,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (3,0),(3,1),(3,2) = 4x3
    
    # 10. Try a different structure: use 4 lines of 3 points each that intersect nicely
    # Line 1: y=0, x=0,1,2
    # Line 2: y=1, x=0,1,2
    # Line 3: x=0, y=0,1,2 (shares with lines 1,2)
    # Line 4: x=1, y=0,1,2 (shares)
    # That's just 3x3 = 9 points
    
    # Let me try: 12 points as vertices of a regular 12-gon? No, not integer.
    
    # 11. Try: (0,0),(1,0),(2,0),(3,0), (0,1),(2,1),(4,1), (0,2),(2,2),(4,2), (1,3),(3,3)
    # Check: 12 points
    cand11 = [[0,0],[1,0],[2,0],[3,0], [0,1],[2,1],[4,1], [0,2],[2,2],[4,2], [1,3],[3,3]]
    candidates.append(cand11)
    
    # 12. Try: 3x3 grid + 3 points at (3,0),(3,3),(0,3)
    candidates.append(ag23 + [[3,0],[3,3],[0,3]])
    
    # 13. Try: 3x3 grid + 3 points at (3,0),(3,1),(3,2) = 4x3 (already)
    
    # 14. Try: points (i,j) for i in 0..2, j in 0..2, plus (3,0),(3,1),(3,2)
    # Already covered
    
    # 15. Try a "staircase" or other pattern
    # (0,0),(1,0),(2,0), (0,1),(1,1),(2,1),(3,1), (0,2),(1,2),(2,2),(3,2),(4,2)
    cand15 = [[0,0],[1,0],[2,0], [0,1],[1,1],[2,1],[3,1], [0,2],[1,2],[2,2],[3,2],[4,2]]
    candidates.append(cand15)
    
    # 16. Try: (0,0),(1,0),(2,0),(3,0), (0,1),(1,1),(2,1),(3,1), (0,2),(1,2),(2,2),(3,2) = 4x3
    
    # 17. Try: 6 points on y=0, 6 points on y=1
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 18. Try: 4 points on each of 3 parallel lines, offset
    # y=0: x=0,1,2,3
    # y=1: x=1,2,3,4
    # y=2: x=2,3,4,5
    cand18 = [[0,0],[1,0],[2,0],[3,0], [1,1],[2,1],[3,1],[4,1], [2,2],[3,2],[4,2],[5,2]]
    candidates.append(cand18)
    
    # 19. Try: 4 points on each of 3 parallel lines, no offset
    # Already 4x3
    
    # 20. Try: 3 points on 4 parallel lines
    # y=0: x=0,1,2
    # y=1: x=0,1,2
    # y=2: x=0,1,2
    # y=3: x=0,1,2
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Now evaluate all candidates
    best_config = None
    best_count = float('inf')
    
    for config in candidates:
        # Check distinctness
        if len(set(map(tuple, config))) != 12:
            continue
        # Check bounds
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in config):
            continue
        # Check not all collinear
        if all_collinear(config):
            continue
        c = count_two_point_lines(config)
        if c < best_count:
            best_count = c
            best_config = config
    
    if best_config is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_config

# EVOLVE-BLOCK-END
