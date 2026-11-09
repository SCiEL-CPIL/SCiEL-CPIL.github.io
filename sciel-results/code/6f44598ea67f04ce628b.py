# EVOLVE-BLOCK-START
import random
from itertools import combinations
from math import gcd

def count_ordinary_lines(points):
    """Count the number of lines that contain exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2)
            A = y2 - y1
            B = -(x2 - x1)
            C = x2 * y1 - y2 * x1
            
            # Normalize
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
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
    for pts in line_to_points.values():
        if len(pts) == 2:
            ordinary += 1
            
    return ordinary

def local_search(points, max_iter=2000, time_limit=10):
    """Perform local search to minimize ordinary lines."""
    import time
    start_time = time.time()
    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Pre-define a set of candidate moves for a point
    # We'll try moving to nearby integer coordinates
    # Range of coordinates is small, so we can try all points in a box
    # But to be safe, we try random neighbors and specific grid points
    
    n = len(current_points)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            old_x, old_y = current_points[i]
            
            # Generate candidate moves
            # 1. Neighbors in a small box
            candidates = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = old_x + dx, old_y + dy
                    # Check bounds and collision
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    # Check collision with other points
                    collision = False
                    for j in range(n):
                        if j != i and current_points[j][0] == nx and current_points[j][1] == ny:
                            collision = True
                            break
                    if not collision:
                        candidates.append([nx, ny])
            
            # Also try some random jumps
            for _ in range(5):
                nx = random.randint(-20, 20)
                ny = random.randint(-20, 20)
                if nx == old_x and ny == old_y:
                    continue
                collision = False
                for j in range(n):
                    if j != i and current_points[j][0] == nx and current_points[j][1] == ny:
                        collision = True
                        break
                if not collision:
                    candidates.append([nx, ny])
            
            # Try each candidate
            for cand in candidates:
                current_points[i] = cand
                new_score = count_ordinary_lines(current_points)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                    improved = True
                    break # Accept first improvement and move to next point
                else:
                    current_points[i] = [old_x, old_y]
            
            if improved:
                break # Restart the loop if any point moved
            
            if best_score == 6:
                return best_points, best_score

    return best_points, best_score

def construct():
    """
    Construct 12 integer points minimizing the number of ordinary lines.
    """
    best_points = None
    best_score = float('inf')
    
    # Candidate initial configurations
    configs = []
    
    # 1. 4x3 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. Hexagonal-like pattern (approximate)
    # Center (0,0), 6 neighbors, 6 further out
    hex_pts = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), 
               (2,0), (-2,0), (0,2), (0,-2), (2,2)]
    configs.append([list(p) for p in hex_pts])
    
    # 3. Symmetric Cross
    cross_pts = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (2,0), (-2,0), 
                 (0,2), (0,-2), (1,1), (-1,-1), (1,-1)]
    configs.append([list(p) for p in cross_pts])
    
    # 4. Random starts
    random.seed(42)
    for _ in range(10):
        pts = []
        used = set()
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        configs.append(pts)
    
    # Run local search on each config
    for config in configs:
        # Check validity
        if len(set(tuple(p) for p in config)) != 12:
            continue
        if all(p[0] == config[0][0] and p[1] == config[0][1] for p in config):
            continue # All collinear (vertical)
        if all(p[1] == config[0][1] for p in config):
            continue # All collinear (horizontal)
            
        # Quick check if all collinear
        collinear = True
        x0, y0 = config[0]
        x1, y1 = config[1]
        for p in config[2:]:
            if (x1-x0)*(p[1]-y0) != (y1-y0)*(p[0]-x0):
                collinear = False
                break
        if collinear:
            continue

        res_pts, res_score = local_search(config, max_iter=500, time_limit=5)
        if res_score < best_score:
            best_score = res_score
            best_points = res_pts
            
        if best_score == 6:
            break

    if best_points is None:
        # Fallback
        best_points = configs[0]
        
    return best_points
# EVOLVE-BLOCK-END
