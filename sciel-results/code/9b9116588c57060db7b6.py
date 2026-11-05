import random
import time
import math
from itertools import combinations

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0 and B < 0:
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
                
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def try_configuration_4x3_grid():
    """
    Try a 4x3 grid arrangement which has 4 rows of 3 points each.
    This should have few ordinary lines.
    """
    points = [[x, y] for x in range(4) for y in range(3)]
    return points

def try_configuration_pappus_like():
    """
    Try a Pappus-like configuration with multiple 3-point lines.
    """
    # Create 4 groups of 3 collinear points
    # Group 1: horizontal at y=0
    # Group 2: horizontal at y=1
    # Group 3: vertical at x=0
    # Group 4: vertical at x=1
    # But this might create too many intersections
    
    # Alternative: use a more structured approach
    # 3 lines of 4 points each (12 points total)
    # Line 1: y = 0, x = 0,1,2,3
    # Line 2: y = 1, x = 0,1,2,3  
    # Line 3: x = 0, y = 0,1,2,3 - but this overlaps
    
    # Let me try: 4 lines of 3 points
    # Line A: (0,0), (1,0), (2,0)
    # Line B: (0,1), (1,1), (2,1)
    # Line C: (0,2), (1,2), (2,2)
    # Line D: (3,0), (3,1), (3,2)
    # This is just a 4x3 grid
    
    # Better: use diagonal lines too
    points = [
        [0, 0], [1, 0], [2, 0],  # Line 1: y=0
        [0, 1], [1, 1], [2, 1],  # Line 2: y=1
        [0, 2], [1, 2], [2, 2],  # Line 3: y=2
        [3, 0], [3, 1], [3, 2],  # Line 4: x=3
    ]
    return points

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 150s limit, leave 5s margin
    
    # Try known configurations first
    candidates = []
    
    # Configuration 1: 4x3 grid
    config1 = try_configuration_4x3_grid()
    candidates.append(config1)
    
    # Configuration 2: Pappus-like
    config2 = try_configuration_pappus_like()
    candidates.append(config2)
    
    # Evaluate initial candidates
    best_points = None
    best_score = float('inf')
    
    for config in candidates:
        score = count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # If we already have 6, return
    if best_score <= 6:
        return best_points
    
    # Simulated Annealing from best starting point
    points = [p[:] for p in best_points]
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a point to move
        i = random.randint(0, 11)
        
        # Pick two other points to define the target line
        j, k = random.sample([x for x in range(12) if x != i], 2)
        
        xj, yj = points[j]
        xk, yk = points[k]
        
        # Direction vector
        dx = xk - xj
        dy = yk - yj
        
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            continue
        dx //= g
        dy //= g
        
        # Generate candidate t values
        candidates_t = []
        
        # Try small t values
        for t in range(-10, 11):
            candidates_t.append(t)
        
        # Add some random t values
        for _ in range(10):
            if dx != 0:
                max_t = int(10**9 / abs(dx))
            else:
                max_t = 10**9
            
            if dy != 0:
                max_t_y = int(10**9 / abs(dy))
            else:
                max_t_y = 10**9
                
            max_t = min(max_t, max_t_y)
            if max_t < 1:
                max_t = 100
            
            t = random.randint(-max_t, max_t)
            candidates_t.append(t)
        
        found = False
        for t in candidates_t:
            x_new = xj + t * dx
            y_new = yj + t * dy
            
            if abs(x_new) > 10**9 or abs(y_new) > 10**9:
                continue
                
            p_new = [x_new, y_new]
            
            # Check if p_new is distinct from all other points
            is_distinct = True
            for idx, p in enumerate(points):
                if idx == i:
                    continue
                if p[0] == x_new and p[1] == y_new:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
                
            # Try the move
            old_p = points[i]
            points[i] = p_new
            
            new_score = count_ordinary_lines(points)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    if best_score <= 6:
                        return best_points
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < math.exp(-delta / T):
                        current_score = new_score
                    else:
                        points[i] = old_p
                        continue
                else:
                    current_score = new_score
            
            found = True
            break
            
        if not found:
            pass
            
        # Cool down
        T = max(T_min, T * cooling_rate)
        
    return best_points
