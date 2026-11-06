import time
import itertools
import random

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly 2 points from the given list.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Dictionary to store lines. Key: normalized (dx, dy, c) or similar.
    # Since points are integers, we can use (dx, dy) and offset.
    # Line through (x1, y1) and (x2, y2):
    # dx = x2 - x1, dy = y2 - y1
    # Normalize: divide by gcd(dx, dy). Ensure sign consistency (e.g., dx > 0 or dx==0, dy>0).
    # c = x1*dy - y1*dx (from dx*y - dy*x = c)
    
    lines = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            g = __import__('math').gcd(abs(dx), abs(dy))
            if g != 0:
                dx //= g
                dy //= g
            
            # Standardize sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Calculate c = x1*dy - y1*dx
            c = x1 * dy - y1 * dx
            
            key = (dx, dy, c)
            
            if key in lines:
                lines[key].add(i)
            else:
                lines[key] = {i, j}
                
    ordinary_count = 0
    for pts_on_line in lines.values():
        if len(pts_on_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initial candidates
    candidates = []
    
    # 3x4 Grid
    grid_34 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_34)
    
    # 4x3 Grid
    grid_43 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_43)
    
    # 2x6 Grid
    grid_26 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(grid_26)
    
    # 6x2 Grid
    grid_62 = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(grid_62)
    
    # Square 3x3 + 3 points?
    # Just use the grids for now.
    
    best_points = candidates[0]
    best_score = count_ordinary_lines(best_points)
    
    # Local Search
    # We will try to move individual points to nearby coordinates.
    # To avoid getting stuck, we can try multiple random restarts or perturbations.
    
    # Define a search space for coordinates, e.g., -5 to 5
    search_range = 5
    
    def get_neighbors(p):
        x, y = p
        neighbors = []
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                # Keep coordinates reasonable to avoid overflow or huge numbers
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    neighbors.append([nx, ny])
        return neighbors

    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simple hill climbing with random restarts
    for restart in range(10): # Try a few restarts
        if restart > 0:
            # Perturb the current best
            current_points = [p[:] for p in best_points]
            # Move 2 random points randomly
            for _ in range(2):
                idx = random.randint(0, 11)
                current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
            current_score = count_ordinary_lines(current_points)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for i in range(12):
                orig_p = current_points[i]
                # Sample a few neighbors to save time
                neighbors = get_neighbors(orig_p)
                random.shuffle(neighbors)
                
                for new_p in neighbors[:20]: # Limit neighbors checked per point
                    # Check if new point is distinct from others
                    if any(p == new_p for p in current_points if current_points.index(p) != i):
                        continue
                    
                    old_p = current_points[i]
                    current_points[i] = new_p
                    
                    new_score = count_ordinary_lines(current_points)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # If this is the best so far, update global best
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        break # Restart neighbor loop for this point? Or continue?
                    else:
                        current_points[i] = old_p
                        
                if time.time() > deadline:
                    break
                    
        if time.time() > deadline:
            break
            
    # Final check
    if time.time() > deadline - 5:
        return best_points
        
    return best_points
