# EVOLVE-BLOCK-START
import math
import random
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            g = math.gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            c = dy_n * x1 - dx_n * y1
            key = (dx_n, dy_n, c)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts in lines.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in points):
        return False
    # Check not all collinear
    (x0, y0) = points[0]
    (x1, y1) = points[1]
    (x2, y2) = points[2]
    if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
        all_collinear = True
        for i in range(3, 12):
            (x3, y3) = points[i]
            if (x1 - x0) * (y3 - y0) != (y1 - y0) * (x3 - x0):
                all_collinear = False
                break
        if all_collinear:
            return False
    return True

def construct():
    best_points = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115
    
    # Greedy Construction with Line Packing
    # We will run multiple iterations with different random seeds/initializations
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with 3 random non-collinear points
        # Use a small range for initial points to keep numbers manageable
        init_points = []
        while len(init_points) < 3:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = [x, y]
            if p not in init_points:
                init_points.append(p)
        
        # Ensure not collinear
        (x0, y0), (x1, y1), (x2, y2) = init_points
        if (x1 - x0) * (y2 - y0) == (y1 - y0) * (x2 - x0):
            continue
            
        current_points = [p[:] for p in init_points]
        current_score = count_ordinary_lines(current_points)
        
        # Greedy addition
        for step in range(3, 12):
            if time.time() - start_time > time_limit:
                break
                
            # Generate candidates
            candidates = set()
            
            # 1. Extend existing lines
            # For every pair of points, consider points on the line defined by them
            # We only need to check a few integer points along the line
            for i in range(len(current_points)):
                for j in range(i + 1, len(current_points)):
                    x1, y1 = current_points[i]
                    x2, y2 = current_points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    
                    if dx == 0 and dy == 0:
                        continue
                    
                    g = math.gcd(abs(dx), abs(dy))
                    dx_n = dx // g
                    dy_n = dy // g
                    
                    # Normalize direction
                    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                        dx_n = -dx_n
                        dy_n = -dy_n
                    
                    # Line equation: dy_n * x - dx_n * y = c
                    c = dy_n * x1 - dx_n * y1
                    
                    # Generate points on this line
                    # We look for integer points (x, y) such that dy_n * x - dx_n * y = c
                    # and they are close to the existing points to avoid huge coordinates
                    # Parametric: x = x1 + t * dx_n, y = y1 + t * dy_n
                    # t is integer
                    # We try t values that put the point in a reasonable range
                    # or near the convex hull
                    
                    # Base point is (x1, y1) at t=0, (x2, y2) at t=dx/dx_n
                    t_max = max(abs(x1), abs(y1), abs(x2), abs(y2)) + 20
                    for t in range(-10, 11):
                        nx = x1 + t * dx_n
                        ny = y1 + t * dy_n
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        # Check if this point is already in current_points
                        if any(nx == p[0] and ny == p[1] for p in current_points):
                            continue
                        candidates.add((nx, ny))
                        
            # 2. Random candidates near the cluster
            # Find centroid
            cx = sum(p[0] for p in current_points) // len(current_points)
            cy = sum(p[1] for p in current_points) // len(current_points)
            
            for _ in range(50):
                nx = cx + random.randint(-5, 5)
                ny = cy + random.randint(-5, 5)
                if any(nx == p[0] and ny == p[1] for p in current_points):
                    continue
                candidates.add((nx, ny))
                
            # Evaluate candidates
            best_candidate = None
            best_candidate_score = float('inf')
            best_candidate_potential = 0
            
            for cand in candidates:
                new_points = [p[:] for p in current_points]
                new_points.append(list(cand))
                
                if not is_valid(new_points):
                    continue
                    
                score = count_ordinary_lines(new_points)
                
                # Potential: number of lines this point lies on (excluding the new ones formed by it alone? No, just count collinearities with existing points)
                # A point on a line with k existing points is good.
                collinearity_count = 0
                for i in range(len(current_points)):
                    for j in range(i + 1, len(current_points)):
                        x1, y1 = current_points[i]
                        x2, y2 = current_points[j]
                        # Check if cand is on line x1x2
                        # (y2-y1)*(x_c - x1) == (x2-x1)*(y_c - y1)
                        if (y2 - y1) * (cand[0] - x1) == (x2 - x1) * (cand[1] - y1):
                            collinearity_count += 1
                            
                # We prefer lower score, and higher collinearity_count as tie-breaker
                if score < best_candidate_score or (score == best_candidate_score and collinearity_count > best_candidate_potential):
                    best_candidate_score = score
                    best_candidate = cand
                    best_candidate_potential = collinearity_count
                    
            if best_candidate is None:
                # Fallback: random point
                while True:
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    if any(nx == p[0] and ny == p[1] for p in current_points):
                        continue
                    new_points = [p[:] for p in current_points]
                    new_points.append([nx, ny])
                    if is_valid(new_points):
                        best_candidate = [nx, ny]
                        best_candidate_score = count_ordinary_lines(new_points)
                        break
            
            current_points.append(best_candidate)
            current_score = best_candidate_score
            
            # Early stopping if we hit the lower bound
            if current_score == 6:
                break
                
        # Final check and local refinement (light)
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            break
            
    if best_points is None:
        # Fallback to a known good configuration if greedy fails
        # 3x4 grid is 12 points.
        best_points = [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
