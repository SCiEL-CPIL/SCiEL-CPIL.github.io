import time
import random
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Dictionary to store lines. Key: (A, B, C) normalized line coefficients.
    # Line equation: Ax + By + C = 0
    # Normalize (A, B, C) such that gcd(A, B, C) = 1 and A > 0 (or A=0, B>0, etc.)
    
    lines = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen
            
            g = math.gcd(math.gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure consistent sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set([i, j])
            else:
                lines[key].add(i)
                lines[key].add(j)
                
    ordinary_count = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Pool of points
    # Use a small grid to keep computations fast and likely to find integer solutions
    # The optimal configuration for 12 points is known to exist with small coordinates.
    # Let's use [-10, 10]
    pool = [[x, y] for x in range(-10, 11) for y in range(-10, 11)]
    
    best_score = float('inf')
    best_points = None
    
    # Initial candidates
    # 1. Random
    # 2. Structured (3x4 grid centered)
    
    random.seed(42)
    
    # Try a few initial configurations
    initial_sets = []
    
    # 3x4 Grid
    grid_points = [[x, y] for x in range(-1, 3) for y in range(-1, 2)]
    initial_sets.append(grid_points)
    
    # Random sets
    for _ in range(10):
        initial_sets.append(random.sample(pool, 12))
        
    # Also try Pappus-like
    # Pappus points on y=0 and y=1
    pappus = [
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
        [-2, 2], [2, 2], [0, 3] # Approximate Pappus line?
    ]
    # Let's just rely on random + grid + local search
    
    for init_points in initial_sets:
        if time.time() > deadline:
            break
        
        # Ensure distinct and valid
        init_points = [list(p) for p in init_points]
        if len(set(map(tuple, init_points))) != 12:
            continue
            
        current_points = init_points
        current_score = count_ordinary_lines(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        if best_score == 6:
            return best_points
            
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for i in range(12):
                if time.time() > deadline:
                    break
                
                old_point = current_points[i]
                
                # Try replacing with a random point from pool
                # Or try all points in pool? 441 * 12 = 5292 evaluations.
                # Each eval is ~1ms? 5 seconds per iteration. Too slow for many iterations.
                # Let's try a subset or random sampling of the pool.
                
                # Sample 50 random points from pool to try
                candidates = random.sample(pool, 50)
                
                best_move = None
                best_move_score = current_score
                
                for new_point in candidates:
                    # Skip if same point
                    if new_point[0] == old_point[0] and new_point[1] == old_point[1]:
                        continue
                    
                    # Check if new_point is already in current_points (other than i)
                    is_duplicate = False
                    for j in range(12):
                        if j != i:
                            if current_points[j][0] == new_point[0] and current_points[j][1] == new_point[1]:
                                is_duplicate = True
                                break
                    if is_duplicate:
                        continue
                        
                    # Create new set
                    new_points = current_points[:]
                    new_points[i] = new_point
                    
                    new_score = count_ordinary_lines(new_points)
                    
                    if new_score < best_move_score:
                        best_move_score = new_score
                        best_move = new_point
                        
                if best_move is not None:
                    current_points[i] = best_move
                    current_score = best_move_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                        
                    if best_score == 6:
                        return best_points
                        
    return best_points
