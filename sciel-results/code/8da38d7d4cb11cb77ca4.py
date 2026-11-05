"""
Minimize ordinary lines for 12 integer points.
Strategy:
1. Fast O(n^3) line counting using normalized line keys.
2. Structured constructive search:
   - Start with known low-ordinary-line configurations (3x3 grid, PG(2,3) projections).
   - Use a "line-completion" heuristic: try adding/moving points to complete lines 
     with existing points, specifically targeting the elimination of ordinary lines.
   - This is a structural change from generic random walks to geometry-aware moves.
"""
import time
import random
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
    
    # Find all unique lines and count points on them
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

def is_valid(pts):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(pts) != 12:
        return False
    # Check distinct
    tuples = [tuple(p) for p in pts]
    if len(set(tuples)) != 12:
        return False
    # Check bounds
    for x, y in tuples:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
    # Check not all collinear
    if len(set(tuples)) >= 3:
        x1, y1 = tuples[0]
        x2, y2 = tuples[1]
        dx = x2 - x1
        dy = y2 - y1
        for i in range(2, 12):
            xi, yi = tuples[i]
            if (xi - x1) * dy - (yi - y1) * dx != 0:
                return True
        return False
    return False

def construct():
    start_time = time.time()
    deadline = time.time() + 145.0  # 5s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(pts, cnt):
        nonlocal best_points, best_count
        if cnt < best_count:
            best_count = cnt
            best_points = [p[:] for p in pts]

    # Strategy 1: Initialize with strong geometric seeds
    # 3x3 grid (9 points) + 3 more
    # The 3x3 grid has 8 lines with 3 points, 0 ordinary lines among the 9?
    # Let's check: 3x3 grid points (0,0)..(2,2).
    # Lines with 3: 3 rows, 3 cols, 2 diags. Total 8 lines.
    # Pairs: C(9,2)=36. Lines with 3 pts cover 3*8=24 pairs. Remaining 12 pairs form 4 ordinary lines?
    # No, the remaining pairs are (0,0)-(2,1), (0,1)-(2,0), (0,2)-(1,0), (1,2)-(0,0)... wait.
    # Actually, 3x3 grid has 4 ordinary lines? Let's just compute.
    
    seed_3x3 = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 points to complete lines.
    # Extend 3x3 to 3x4 or 4x3?
    # 4x3 grid:
    grid_43 = [[x, y] for x in range(4) for y in range(3)]
    if is_valid(grid_43):
        cnt = count_ordinary_lines(grid_43)
        update_best(grid_43, cnt)
        
    grid_34 = [[x, y] for x in range(3) for y in range(4)]
    if is_valid(grid_34):
        cnt = count_ordinary_lines(grid_34)
        update_best(grid_34, cnt)
        
    # Pappus configuration (9 points, 9 lines of 3)
    # Projective plane PG(2,3) has 13 points, 13 lines of 4.
    # We can take 12 of them.
    # Let's construct PG(2,3) points in affine plane + infinity.
    # Affine points: (x,y) in F3^2. 9 points.
    # Lines in F3: y = mx + c. 3*3=9 lines.
    # Plus 1 line at infinity.
    # In PG(2,3), each line has 4 points.
    # If we take all 13 points, ordinary lines = 0.
    # We need 12 points. Remove 1 point.
    # The remaining 12 points:
    # Lines that didn't contain the removed point still have 4 points.
    # Lines that contained the removed point now have 3 points.
    # So ordinary lines should be 0.
    # Can we embed PG(2,3) in integer coordinates?
    # Yes, using a standard rational parametrization or just mapping F3 to integers 0,1,2 and handling infinity carefully.
    # However, the "line at infinity" in projective plane corresponds to parallel lines in affine.
    # In integer grid, parallel lines don't meet.
    # But we can use a rational representation.
    # Let's try to generate PG(2,3) points in Z^2.
    # Points: (x,y) for x,y in {0,1,2}.
    # Lines with slope m in {0,1,2} and intercept c in {0,1,2}.
    # y = mx + c (mod 3).
    # We need to map this to integers such that collinearity is preserved.
    # This is tricky because mod 3 arithmetic doesn't map directly to standard integer lines without scaling.
    # Alternative: Use the 3x3 grid and add points at "infinity" by scaling?
    # No, just use the 4x3 grid as a strong seed.
    
    # Strategy 2: Structural Local Search (Line Completion)
    # Instead of random moves, identify ordinary lines and try to "fix" them by moving a point 
    # to a position that creates a new 3-point line or joins an existing line.
    
    def structural_local_search(pts, max_iter=5000):
        nonlocal best_points, best_count
        cur = [p[:] for p in pts]
        cur_cnt = count_ordinary_lines(cur)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # Identify ordinary lines
            def get_ordinary_lines(points):
                n = len(points)
                lines = {}
                def line_key(p1, p2):
                    x1, y1 = p1
                    x2, y2 = p2
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        return None
                    if dx < 0 or (dx == 0 and dy < 0):
                        dx = -dx
                        dy = -dy
                    g = math.gcd(abs(dx), abs(dy))
                    if g > 0:
                        dx //= g
                        dy //= g
                    c = dy * x1 - dx * y1
                    return (dx, dy, c)
                    
                for i in range(n):
                    for j in range(i + 1, n):
                        key = line_key(points[i], points[j])
                        if key is not None:
                            if key not in lines:
                                lines[key] = set()
                            lines[key].add(i)
                            lines[key].add(j)
                            
                ordinary = []
                for key, pts_on_line in lines.items():
                    if len(pts_on_line) == 2:
                        ordinary.append(list(pts_on_line))
                return ordinary
            
            ordinary_lines = get_ordinary_lines(cur)
            if not ordinary_lines:
                break
                
            # Pick a random ordinary line
            i, j = random.choice(ordinary_lines)
            
            # Try to move point i or j to eliminate this ordinary line
            # Move point i to a new location that creates a 3-point line with some other point k
            moved = False
            for _ in range(100):
                if moved:
                    break
                # Try to place point i on a line defined by two other points
                # Or just try random moves but biased towards existing points
                # Let's try moving i to a position that is collinear with two other random points
                k, l = random.sample([x for x in range(12) if x not in [i, j]], 2)
                p1 = cur[k]
                p2 = cur[l]
                # The line through p1 and p2. We want to place i on this line.
                # But i is already a point. We want to move i to a NEW point on this line?
                # No, i is fixed in the set, we move its coordinates.
                # If we move i to be on the line of k and l, then the line (k,l) becomes a 3-point line.
                # This might eliminate the ordinary line (i,j) if i moves off it.
                
                # Calculate a new position for i on the line k-l
                # Line equation: A x + B y + C = 0
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                # Direction vector
                # Normal vector: (dy, -dx)
                # Point on line: p1
                # We want a new integer point (x, y) on this line, different from k, l, and other points.
                # Parametric: (x1 + t*dx, y1 + t*dy)
                # Try small t values
                for t in range(-10, 11):
                    if t == 0: # p1
                        continue
                    if t*dx == dx and t*dy == dy: # p2 (if t=1)
                        continue
                    new_x = x1 + t * dx
                    new_y = y1 + t * dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    new_pt = [new_x, new_y]
                    # Check if new_pt is already in the set (excluding i)
                    valid = True
                    for m in range(12):
                        if m == i:
                            continue
                        if cur[m][0] == new_x and cur[m][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    old_i = cur[i]
                    cur[i] = new_pt
                    if not is_valid(cur):
                        cur[i] = old_i
                        continue
                    new_cnt = count_ordinary_lines(cur)
                    if new_cnt < cur_cnt:
                        cur_cnt = new_cnt
                        improved = True
                        update_best(cur, cur_cnt)
                        if best_count <= 6:
                            return cur
                        moved = True
                        break
                    else:
                        cur[i] = old_i
            
            # If no improvement, try a small random kick
            if not improved:
                i = random.randint(0, 11)
                old_x, old_y = cur[i]
                for _ in range(10):
                    dx = random.randint(-2, 2)
                    dy = random.randint(-2, 2)
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = old_x + dx, old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    cur[i] = [new_x, new_y]
                    if not is_valid(cur):
                        cur[i] = [old_x, old_y]
                        continue
                    new_cnt = count_ordinary_lines(cur)
                    if new_cnt < cur_cnt:
                        cur_cnt = new_cnt
                        improved = True
                        update_best(cur, cur_cnt)
                        break
                    else:
                        cur[i] = [old_x, old_y]
                        
        return cur

    # Run structural search from best seed
    if best_points is not None:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            structural_local_search(best_points, max_iter=2000)
            if best_count <= 6:
                break

    # Strategy 3: Try specific PG(2,3) subset
    # Points in PG(2,3) can be represented as homogeneous coordinates [x:y:z] over F3.
    # We can map to affine coordinates (x/z, y/z) for z!=0, and points at infinity.
    # To embed in Z^2, we can use a large scaling factor or just pick a specific representation.
    # Let's try the 4x3 grid again, it's often very good.
    
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
