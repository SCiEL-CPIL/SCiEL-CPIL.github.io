import time
import random
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Count lines containing exactly 2 of the given points.
    Uses a dictionary to map line keys to point indices.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map line key to set of point indices
    line_map = defaultdict(set)
    
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            
            # Normalize direction vector
            if dx == 0 and dy == 0:
                continue
                
            # GCD normalization
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Canonical sign
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: dy_n * x - dx_n * y = c
            c = dy_n * xi - dx_n * yi
            line_key = (dx_n, dy_n, c)
            
            line_map[line_key].add(i)
            line_map[line_key].add(j)
            
    ordinary = 0
    for key, pts in line_map.items():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configurations
    # 1. 4x4 Grid minus corners (Candidate for low ordinary lines)
    # 2. 3x4 Grid
    # 3. Random points
    
    candidates = []
    
    # 4x4 grid minus corners
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    # Remove corners (0,0), (0,3), (3,0), (3,3)
    # Indices: (0,0)=0, (0,3)=3, (3,0)=12, (3,3)=15
    # Wait, range(4) for x, range(4) for y.
    # (0,0) is index 0. (0,3) is index 3. (3,0) is index 12. (3,3) is index 15.
    rem_indices = {0, 3, 12, 15}
    config_1 = [p for i, p in enumerate(grid_4x4) if i not in rem_indices]
    candidates.append(config_1)
    
    # 3x4 Grid
    config_2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(config_2)
    
    # 4x3 Grid
    config_3 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(config_3)
    
    # Random configurations
    random.seed(42)
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if [x, y] not in pts:
                pts.append([x, y])
        candidates.append(pts)
        
    best_points = None
    best_score = float('inf')
    
    for config in candidates:
        score = count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
            
    # Local Search
    # We will try to move points to reduce the score.
    # To speed up, we can limit the search space.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            x0, y0 = best_points[i]
            # Try nearby moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    
                    new_points = [list(p) for p in best_points]
                    new_points[i] = [nx, ny]
                    
                    # Check distinctness
                    if len(set(tuple(p) for p in new_points)) != 12:
                        continue
                    
                    # Check not all collinear
                    # Simple check: not all on same line as first two
                    if all((new_points[j][0] - new_points[0][0]) * (new_points[1][1] - new_points[0][1]) == 
                           (new_points[1][0] - new_points[0][0]) * (new_points[j][1] - new_points[0][1]) 
                           for j in range(12)):
                        continue
                        
                    score = count_ordinary_lines(new_points)
                    if score < best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        if best_score == 6: # Theoretical min
                            return [list(p) for p in best_points]
                        
    # If local search didn't find 6, try random restarts with larger jumps
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Randomize one point significantly
            i = random.randint(0, 11)
            new_x = random.randint(-10, 10)
            new_y = random.randint(-10, 10)
            new_points = [list(p) for p in best_points]
            new_points[i] = [new_x, new_y]
            
            if len(set(tuple(p) for p in new_points)) != 12:
                continue
                
            score = count_ordinary_lines(new_points)
            if score < best_score:
                best_score = score
                best_points = new_points
                
                # Quick local search
                local_improved = True
                while local_improved and time.time() < deadline:
                    local_improved = False
                    for j in range(12):
                        if time.time() > deadline:
                            break
                        x0, y0 = best_points[j]
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x0 + dx, y0 + dy
                                if abs(nx) > 10**9 or abs(ny) > 10**9:
                                    continue
                                temp_points = [list(p) for p in best_points]
                                temp_points[j] = [nx, ny]
                                if len(set(tuple(p) for p in temp_points)) != 12:
                                    continue
                                s = count_ordinary_lines(temp_points)
                                if s < best_score:
                                    best_score = s
                                    best_points = temp_points
                                    local_improved = True
                                    if best_score == 6:
                                        return [list(p) for p in best_points]

    return [list(p) for p in best_points]
