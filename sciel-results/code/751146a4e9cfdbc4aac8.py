"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def count_ordinary_lines_fast(points):
    """
    Count lines containing exactly 2 of the given points.
    Uses O(n^3) logic: for each pair, determine the line, then check how many points lie on it.
    To optimize, we cache line counts or use a set of lines.
    For n=12, O(n^3) is 1728 operations per call, which is very fast.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line representation to count of points on that line
    # Line representation: (dy, dx, c) where dy*x - dx*y = c
    # dx, dy are normalized such that gcd(dx,dy)=1 and dx>0 or (dx==0 and dy>0)
    # c is determined by the first point.
    
    line_counts = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize direction
        if dx == 0 and dy == 0:
            return None
        
        # Ensure consistent orientation
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        # Calculate c = dy*x1 - dx*y1
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def point_on_line(p, line_key):
        if line_key is None:
            return False
        dx, dy, c = line_key
        x, y = p
        return dy * x - dx * y == c

    # Iterate over all pairs to identify unique lines
    # Then for each unique line, count points.
    # Since n is small (12), we can just do this efficiently.
    
    # 1. Identify all unique lines defined by pairs
    unique_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                unique_lines.add(key)
                
    # 2. For each unique line, count how many points lie on it
    ordinary_count = 0
    for line_key in unique_lines:
        count = 0
        for p in points:
            if point_on_line(p, line_key):
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            
    # 1. Try the 3x4 grid as baseline
    grid_points = [[x, y] for x in range(4) for y in range(3)]
    grid_count = count_ordinary_lines_fast(grid_points)
    update_best(grid_points, grid_count)
    
    # 2. Structured Search: Pappus-like configurations
    # The Pappus configuration has 9 points on 9 lines of 3.
    # We need 12 points.
    # Strategy: Start with 3 lines of 4 points (12 points total).
    # If we can arrange 12 points on 3 lines such that the intersections create more collinearities,
    # we might reduce ordinary lines.
    
    # However, 3 lines of 4 points generally have many ordinary lines unless the lines are concurrent or parallel in specific ways.
    # If 3 lines are concurrent, they share 1 point. Total points = 3*4 - 2 = 10 (if all distinct intersections are just the center).
    # Wait, if 3 lines are concurrent at P, and each has 4 points including P, total points = 3*3 + 1 = 10.
    # We need 12 points.
    
    # Let's try: 4 lines of 3 points?
    # If we use a grid, we get many horizontal/vertical/diagonal lines.
    
    # Let's use a faster local search with the fast counter.
    # We can afford many more iterations now.
    
    def local_improve(points):
        pts = [p[:] for p in points]
        current_count = count_ordinary_lines_fast(pts)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(12):
                # Try moving point i to nearby integer coordinates
                # To be efficient, we can sample a few moves or use a grid around current pos
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        old_x, old_y = pts[i]
                        new_x, new_y = old_x + dx, old_y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        pts[i] = [new_x, new_y]
                        if len(set(tuple(p) for p in pts)) < 12:
                            pts[i] = [old_x, old_y]
                            continue
                        cnt = count_ordinary_lines_fast(pts)
                        if cnt < current_count:
                            current_count = cnt
                            improved = True
                        else:
                            pts[i] = [old_x, old_y]
        return pts, current_count

    random.seed(42)
    
    # Strategy A: Random restarts with local improvement
    # Since counting is fast, we can do more restarts.
    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random points in a small box to encourage collinearities
        # Box size 10x10 gives 100 possible spots, 12 points -> high chance of collisions/collinearities
        pts = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
        
        # Ensure distinct
        if len(set(tuple(p) for p in pts)) < 12:
            # Retry with larger box or just skip
            continue
            
        cnt = count_ordinary_lines_fast(pts)
        update_best(pts, cnt)
        
        # Local improvement
        if cnt <= best_count + 2:
            improved_pts, improved_cnt = local_improve(pts)
            update_best(improved_pts, improved_cnt)
            
            # If we hit 6, we can stop early if we want, but let's keep searching for robustness
            # Actually, 6 is the theoretical minimum. If we find it, we can return immediately.
            if best_count == 6:
                return best_points

    # Strategy B: Specific geometric constructions
    # Try to construct points on a few lines.
    # Example: 3 lines with 4 points each, arranged to maximize intersections.
    # Let lines be y=0, y=1, y=2.
    # Points: (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # This is the 3x4 grid.
    
    # What if we shift some points?
    # Try a "staircase" or "zigzag" on 3 lines.
    
    # Let's try a deterministic search over small integer coordinates for the first few points
    # and solve for the rest? No, too complex.
    
    # Let's try to improve the best found so far with a more aggressive local search
    if best_points:
        # Use a larger step size for the final polish
        def aggressive_local_improve(points):
            pts = [p[:] for p in points]
            current_count = count_ordinary_lines_fast(pts)
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for i in range(12):
                    for dx in range(-5, 6):
                        for dy in range(-5, 6):
                            if dx == 0 and dy == 0:
                                continue
                            old_x, old_y = pts[i]
                            new_x, new_y = old_x + dx, old_y + dy
                            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                continue
                            pts[i] = [new_x, new_y]
                            if len(set(tuple(p) for p in pts)) < 12:
                                pts[i] = [old_x, old_y]
                                continue
                            cnt = count_ordinary_lines_fast(pts)
                            if cnt < current_count:
                                current_count = cnt
                                improved = True
                            else:
                                pts[i] = [old_x, old_y]
            return pts, current_count
            
        improved_pts, improved_cnt = aggressive_local_improve(best_points)
        update_best(improved_pts, improved_cnt)

    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
