"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map line (defined by normalized equation) to set of points on it
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            # Points are distinct, so g != 0
            if g == 0:
                continue
            g = abs(g)
            
            dx_n = dx // g
            dy_n = dy // g
            
            # Normalize direction to ensure consistent key
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: dy_n * x - dx_n * y = c
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add((x1, y1))
            line_points[key].add((x2, y2))
            
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _delta_score(points, idx, new_pt, line_cache):
    """
    Compute the change in ordinary line count when moving point idx to new_pt.
    points: list of 12 points
    idx: index of point to move
    new_pt: new position [x, y]
    line_cache: dict mapping line_key -> set of point indices on that line
    Returns the delta (new_score - old_score)
    """
    n = len(points)
    old_x, old_y = points[idx]
    if new_pt[0] == old_x and new_pt[1] == old_y:
        return 0
    
    # First, check if new_pt is already occupied
    for k, p in enumerate(points):
        if k != idx and p[0] == new_pt[0] and p[1] == new_pt[1]:
            return float('inf')  # Invalid move
    
    # For each line through old point and another point, count how many points are on it
    # For each line through new point and another point, count how many points are on it
    
    # Count effect of removing old point
    # For each other point j, the line (old, j) may lose old point
    # The line (old, j) might still have other points
    
    # Count effect of adding new point
    # For each other point j, the line (new, j) may gain new point
    
    # This is complex. Let's just recompute from scratch but optimized.
    # Actually, for 12 points, recomputing is O(66) which is fast.
    # Let's just call _count_ordinary_lines directly for simplicity, 
    # but we need to be careful about time.
    
    # Alternative: use the line_cache to compute delta more efficiently
    # But for now, let's just recompute. It's 66 pairs, very fast.
    
    new_points = points.copy()
    new_points[idx] = [new_pt[0], new_pt[1]]
    return _count_ordinary_lines(new_points)

def _local_search(points, max_iterations=5000, time_limit=10.0):
    """
    Local search to minimize ordinary lines.
    """
    start_time = time.time()
    best_points = [p[:] for p in points]
    best_score = _count_ordinary_lines(best_points)
    
    current_temp = 1.0
    cooling_rate = 0.99
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = best_points[idx]
        
        # Generate candidate moves: random nearby points
        # Try several random offsets
        found_improvement = False
        for _ in range(20):  # Try 20 random moves per point
            # Random offset in a reasonable range
            radius = random.choice([1, 2, 3, 5, 10])
            new_x = old_x + random.randint(-radius, radius)
            new_y = old_y + random.randint(-radius, radius)
            
            # Check bounds
            if not (-10**9 <= new_x <= 10**9 and -10**9 <= new_y <= 10**9):
                continue
            
            # Check distinctness
            new_pt = [new_x, new_y]
            is_distinct = True
            for k, p in enumerate(best_points):
                if k == idx:
                    continue
                if p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
            
            # Calculate new score
            new_points = best_points.copy()
            new_points[idx] = new_pt
            new_score = _count_ordinary_lines(new_points)
            
            delta = new_score - best_score
            
            if delta < 0:
                best_points = new_points
                best_score = new_score
                found_improvement = True
                break
            elif delta == 0 and random.random() < 0.3:
                best_points = new_points
                # best_score unchanged
        
        if not found_improvement:
            # No improvement found, cool down
            current_temp *= cooling_rate
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try multiple initial configurations and local search
    # Configuration 1: 3x4 grid
    configs = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid (transposed)
        [[x, y] for x in range(3) for y in range(4)],
        # Two parallel lines with 6 points each
        [[x, 0] for x in range(6)] + [[x, 1] for x in range(6)],
        # Three lines with 4 points each
        [[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)],
        # Regular hexagon-like structure
        [[2, 0], [1, 1], [-1, 1], [-2, 0], [-1, -1], [1, -1],
         [4, 0], [2, 2], [-2, 2], [-4, 0], [-2, -2], [2, -2]],
        # Two rectangles
        [[0, 0], [3, 0], [3, 2], [0, 2],
         [5, 0], [8, 0], [8, 2], [5, 2],
         [1, 4], [4, 4], [4, 6], [1, 6]],
    ]
    
    # Also generate random configurations
    for _ in range(10):
        pts = []
        used = set()
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        configs.append(pts)
    
    best_points = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Local search from this configuration
        improved_points, score = _local_search(config, max_iterations=2000, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_points = improved_points
        
        if best_score <= 6:
            break
    
    # Additional global search if time permits
    if time.time() < deadline - 30 and best_score > 6:
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Random restart with local search
            random_pts = []
            used = set()
            while len(random_pts) < 12:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in used:
                    used.add((x, y))
                    random_pts.append([x, y])
            
            improved_points, score = _local_search(random_pts, max_iterations=3000, time_limit=10.0)
            
            if score < best_score:
                best_score = score
                best_points = improved_points
            
            if best_score <= 6:
                break
    
    return best_points
# EVOLVE-BLOCK-END
