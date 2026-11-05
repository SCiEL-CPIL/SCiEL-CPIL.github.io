"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import gcd
import random

def count_two_point_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    two_point_lines = sum(1 for pts in line_to_points.values() if len(pts) == 2)
    return two_point_lines

def all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return False
    return True

def construct():
    """
    Construct 12 points minimizing the number of 2-point lines.
    Uses local search to improve a random initial configuration.
    """
    
    N = 12
    best_score = float('inf')
    best_config = None
    
    random.seed(42)
    
    # Define a search space for points
    # Use a larger grid to allow for more diverse configurations
    grid_limit = 20
    grid_points = set((x, y) for x in range(-grid_limit, grid_limit+1) for y in range(-grid_limit, grid_limit+1))
    
    # Generate initial configurations
    # 1. Random
    # 2. Grid-based (3x4, 2x6, etc.)
    
    candidates = []
    
    # Add 3x4 grid
    for x in range(4):
        for y in range(3):
            candidates.append([x, y])
            
    # Add 4x3 grid
    for x in range(3):
        for y in range(4):
            candidates.append([x, y])
            
    # Add random sets
    for _ in range(50):
        pts = random.sample(list(grid_points), N)
        # Ensure not all collinear
        if not all_collinear(pts):
            candidates.append(pts)
            
    # Local search improvement
    def improve(points):
        points = [tuple(p) for p in points]
        current_score = count_two_point_lines(points)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                # Try moving point i to a new location
                # Candidates: 
                # - Points on existing lines passing through other points
                # - Random points from grid
                # - Points that form 3-point lines with 2 existing points
                
                best_new_point = None
                best_new_score = current_score
                
                # Generate candidate points
                cands = set()
                
                # 1. Intersections of lines defined by pairs of other points
                # This is expensive, so we limit it
                # Instead, we check points on lines defined by pairs of other points
                for j in range(N):
                    if i == j: continue
                    for k in range(j+1, N):
                        if i == k: continue
                        # Line through j and k
                        # Check if there are other points on this line
                        # If we move i to this line, does it help?
                        # We just add the intersection of line(j,k) with some other line?
                        # Simpler: Add points that are close to current point i
                        # Or add points that form a 3-point line with two others.
                        
                        # Let's try adding points that complete a 3-point line with two existing points
                        # For pair (j, k), is there a point p such that j, k, p are collinear and p is in grid?
                        # We can iterate through grid points and check if they form a 3-point line with any pair.
                        
                # Efficient approach:
                # For each pair of other points (j, k), find the line.
                # If we place i on this line, the number of points on this line increases.
                # We want to minimize ordinary lines.
                
                # Let's just try a set of "smart" moves:
                # 1. Move i to a point on an existing line with >= 2 points.
                # 2. Move i to a point that forms a 3-point line with two other points.
                
                # Collect all lines with >= 2 points
                lines = {}
                for j in range(N):
                    if i == j: continue
                    for k in range(j+1, N):
                        if i == k: continue
                        x1, y1 = points[j]
                        x2, y2 = points[k]
                        A = y2 - y1
                        B = x1 - x2
                        C = x2 * y1 - x1 * y2
                        g = gcd(abs(A), gcd(abs(B), abs(C)))
                        if g != 0:
                            A //= g
                            B //= g
                            C //= g
                        if A < 0 or (A == 0 and B < 0):
                            A = -A
                            B = -B
                            C = -C
                        key = (A, B, C)
                        if key not in lines:
                            lines[key] = set()
                        lines[key].add(j)
                        lines[key].add(k)
                
                # Candidates from lines
                for key, pts_on_line in lines.items():
                    if len(pts_on_line) >= 2:
                        # Find integer points on this line in the grid
                        A, B, C = key
                        # Ax + By + C = 0
                        # Iterate x, solve for y
                        # |x|, |y| <= grid_limit
                        for x in range(-grid_limit, grid_limit+1):
                            val = -A*x - C
                            if B == 0:
                                if val == 0:
                                    # Any y works, but we need specific points
                                    # If B=0, A!=0, x is fixed.
                                    # We already have points on this line.
                                    pass
                            else:
                                if val % B == 0:
                                    y = val // B
                                    if -grid_limit <= y <= grid_limit:
                                        p = (x, y)
                                        if p not in [tuple(q) for q in points]:
                                            cands.add(p)
                
                # Add random candidates
                for _ in range(10):
                    p = random.choice(list(grid_points))
                    if p not in [tuple(q) for q in points]:
                        cands.add(p)
                
                # Evaluate candidates
                for p in cands:
                    test_points = list(points)
                    test_points[i] = p
                    if len(set(test_points)) < N:
                        continue
                    if all_collinear(test_points):
                        continue
                    score = count_two_point_lines(test_points)
                    if score < best_new_score:
                        best_new_score = score
                        best_new_point = p
                
                if best_new_point is not None:
                    points[i] = best_new_point
                    current_score = best_new_score
                    improved = True
        
        return [list(p) for p in points], current_score

    for config in candidates:
        if len(config) != N:
            continue
        if all_collinear(config):
            continue
        
        improved_config, score = improve(config)
        
        if score < best_score:
            best_score = score
            best_config = improved_config

    if best_config is None:
        best_config = [[x, y] for x in range(4) for y in range(3)]
        
    return best_config

# EVOLVE-BLOCK-END
