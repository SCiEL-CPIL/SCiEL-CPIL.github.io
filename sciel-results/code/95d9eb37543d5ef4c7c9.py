import time
import math
from itertools import combinations

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
        
        # Normalize direction vector
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        c = dy * x1 - dx * y1
        return (dx, dy, c)
    
    # Find all unique lines
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = line_key(points[i], points[j])
            if key is not None:
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
    
    # Count ordinary lines (exactly 2 points)
    ordinary = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    start_time = time.time()
    deadline = time.time() + 145.0
    
    # Known optimal or near-optimal configurations
    # The theoretical minimum for 12 points is 6.
    # Let's try to construct configurations with 6 ordinary lines.
    
    # One such configuration is the "12-point configuration with 6 ordinary lines"
    # This is related to the Sylvester-Gallai theorem exceptions or specific finite geometries.
    # A known example is a subset of the 3x4 grid or a specific arrangement.
    
    # Let's try a specific known configuration for 12 points with 6 ordinary lines.
    # Reference: "On the number of ordinary lines in a set of points"
    # One construction involves points on 3 lines with 4 points each, plus some adjustments.
    
    # Actually, a 3x4 grid has 12 points.
    # Let's analyze the 3x4 grid:
    # Points: (i, j) for i in 0..2, j in 0..3
    # Lines with 3 or more points:
    # - 3 horizontal lines (y=0,1,2,3? No, x=0,1,2 are verticals with 4 points each? No, i is x, j is y.
    #   If i in 0..2 (3 cols) and j in 0..3 (4 rows):
    #   Vertical lines x=0, x=1, x=2 each have 4 points. (3 lines)
    #   Horizontal lines y=0, y=1, y=2, y=3 each have 3 points. (4 lines)
    #   Diagonals?
    #   Slope 1: (0,0)-(1,1)-(2,2) [3 pts], (0,1)-(1,2) [2 pts], (0,2)-(1,3) [2 pts], (1,0)-(2,1) [2 pts], (1,1)-(2,2) already counted.
    #   Slope -1: (0,3)-(1,2)-(2,1) [3 pts], (0,2)-(1,1)-(2,0) [3 pts], (0,1)-(1,0) [2 pts], (1,3)-(2,2) [2 pts].
    #   Other slopes?
    #   Let's just compute it.
    
    grid_34 = [[x, y] for x in range(3) for y in range(4)]
    best_points = grid_34
    best_count = count_ordinary_lines(grid_34)
    
    # Try other small grids
    grid_43 = [[x, y] for x in range(4) for y in range(3)]
    cnt = count_ordinary_lines(grid_43)
    if cnt < best_count:
        best_count = cnt
        best_points = grid_43
        
    # Try 2x6 grid
    grid_26 = [[x, y] for x in range(2) for y in range(6)]
    cnt = count_ordinary_lines(grid_26)
    if cnt < best_count:
        best_count = cnt
        best_points = grid_26
        
    # Try 6x2 grid
    grid_62 = [[x, y] for x in range(6) for y in range(2)]
    cnt = count_ordinary_lines(grid_62)
    if cnt < best_count:
        best_count = cnt
        best_points = grid_62

    # Try a specific configuration known to have few ordinary lines
    # The "12-point configuration" with 6 ordinary lines is not trivial to find by hand.
    # However, we can use a local search to improve from the best grid.
    
    def local_search(pts, max_iter=5000):
        nonlocal best_points, best_count
        cur = [p[:] for p in pts]
        cur_cnt = count_ordinary_lines(cur)
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            i = int(time.time() * 1000) % 12
            # Try several random moves
            for _ in range(50):
                dx = int((time.time() * 10000) % 11) - 5
                dy = int((time.time() * 10001) % 11) - 5
                if dx == 0 and dy == 0:
                    continue
                old_x, old_y = cur[i]
                new_x, new_y = old_x + dx, old_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                cur[i] = [new_x, new_y]
                # Check distinctness
                valid = True
                for j in range(12):
                    if j != i and cur[j] == cur[i]:
                        valid = False
                        break
                if not valid:
                    cur[i] = [old_x, old_y]
                    continue
                new_cnt = count_ordinary_lines(cur)
                if new_cnt < cur_cnt:
                    cur_cnt = new_cnt
                    improved = True
                    if cur_cnt < best_count:
                        best_count = cur_cnt
                        best_points = [p[:] for p in cur]
                        if best_count <= 6:
                            return cur
                    break
                else:
                    cur[i] = [old_x, old_y]
        return cur
    
    # Run local search from the best grid
    local_search(best_points, max_iter=10000)
    
    if best_count <= 6:
        return best_points
        
    # If we haven't found 6, try a more structured search
    # We know that for 12 points, the minimum is 6.
    # Let's try to construct a configuration with 3 lines of 4 points.
    # If we have 3 lines L1, L2, L3 each with 4 points, and they intersect in 3 distinct points,
    # the total number of points is 4+4+4 - 3 (intersections) = 9. We need 3 more points.
    # This is getting complex. Let's rely on the local search having found a good configuration.
    
    return best_points
