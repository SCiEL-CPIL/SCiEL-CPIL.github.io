import random
import time
import itertools

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line (normalized) to count of points
    # Line representation: (A, B, C) where Ax + By + C = 0
    # To handle integers and avoid floats:
    # For points p1(x1,y1), p2(x2,y2):
    # A = y1 - y2
    # B = x2 - x1
    # C = x1*y2 - x2*y1
    # Normalize (A, B, C) by GCD and sign.
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen
            
            g = 0
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Sign normalization: make first non-zero positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                
    ordinary = 0
    for count in lines.values():
        if count == 2:
            ordinary += 1
            
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    # Cross product
    return (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0) != 0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x3 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    
    best_score = _count_ordinary_lines(points)
    best_points = [p[:] for p in points]
    
    # Local Search
    # Moves:
    # 1. Move a point to a random new integer coordinate.
    # 2. Swap two points? No, points are a set.
    # 3. Move a point to a position that creates a 3-point line with two existing points.
    
    # Heuristic: Try to move points to "interesting" positions.
    # Interesting positions: Intersections of lines defined by pairs of other points.
    
    random.seed(42)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline:
        # Generate candidate moves
        moves = []
        
        # 1. Random moves
        for i in range(12):
            # Move point i to a random location near current
            # Or random in a larger box
            x_new = random.randint(-20, 20)
            y_new = random.randint(-20, 20)
            if [x_new, y_new] not in current_points:
                new_points = [p[:] for p in current_points]
                new_points[i] = [x_new, y_new]
                moves.append(new_points)
        
        # 2. Structured moves: Place point i on a line defined by j, k
        for i in range(12):
            for j in range(12):
                if i == j: continue
                for k in range(12):
                    if i == k or j == k: continue
                    x1, y1 = current_points[j]
                    x2, y2 = current_points[k]
                    # Line through j, k
                    # Try to place i on this line at an integer coordinate
                    # dx = x2-x1, dy = y2-y1
                    dx = x2 - x1
                    dy = y2 - y1
                    # Parametric: x = x1 + t*dx, y = y1 + t*dy
                    # We want integer x, y.
                    # If dx, dy have gcd > 1, we can step by gcd.
                    g = 0
                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a
                    g = gcd(abs(dx), abs(dy))
                    if g == 0: continue
                    step_x = dx // g
                    step_y = dy // g
                    
                    # Try t = -5 to 5
                    for t in range(-5, 6):
                        x_new = x1 + t * step_x
                        y_new = y1 + t * step_y
                        if abs(x_new) <= 10**9 and abs(y_new) <= 10**9:
                            if [x_new, y_new] not in current_points:
                                new_points = [p[:] for p in current_points]
                                new_points[i] = [x_new, y_new]
                                moves.append(new_points)
        
        # Evaluate moves
        best_move_score = current_score
        best_move_points = current_points
        
        # Limit number of moves to evaluate to avoid timeout
        random.shuffle(moves)
        eval_limit = 50
        for m in moves[:eval_limit]:
            if not _not_all_collinear(m):
                continue
            score = _count_ordinary_lines(m)
            if score < best_move_score:
                best_move_score = score
                best_move_points = m
                if best_move_score == 6:
                    break # Found optimal lower bound?
        
        if best_move_score < current_score:
            current_score = best_move_score
            current_points = best_move_points
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
        # If stuck, maybe do a random restart or larger jump
        if time.time() > deadline - 10:
            break
            
    return best_points
