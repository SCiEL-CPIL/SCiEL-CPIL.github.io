import time
from math import gcd
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
    
    ordinary_count = 0
    for key, count in line_counts.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy: 
    # The theoretical minimum for n=12 is 6 ordinary lines.
    # This is achieved by configurations related to the affine plane of order 3 (AG(2,3)) 
    # which has 9 points and 12 lines (each with 3 points), 0 ordinary lines.
    # However, we need 12 points.
    # 
    # A known configuration for 12 points with 6 ordinary lines is derived from 
    # adding 3 points to the AG(2,3) configuration such that they lie on the 
    # "lines at infinity" or similar structures, but since we are in affine space, 
    # we look for a specific integer grid.
    #
    # Actually, a simpler approach: 
    # The "Pappus configuration" has 9 points, 9 lines of 3 points, 0 ordinary lines.
    # If we take a 3x4 grid, we have 12 points.
    # Let's analyze the 3x4 grid more carefully or try to construct a "near" AG(2,3) extension.
    
    # Candidate 1: 3x4 Grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    c1 = count_ordinary_lines(grid_3x4)
    
    # Candidate 2: 4x3 Grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    c2 = count_ordinary_lines(grid_4x3)
    
    # Candidate 3: A specific configuration known to have few ordinary lines.
    # Reference: "On the number of ordinary lines determined by a set of points"
    # For n=12, min is 6.
    # One such configuration is:
    # 3 lines of 4 points, pairwise intersecting, forming a triangle of 3 points,
    # plus 3 additional points? No, that's 9 points.
    #
    # Let's try a configuration based on the "Hesse configuration" (9 points, 12 lines of 3).
    # The Hesse configuration is the set of inflection points of a cubic curve.
    # It has 9 points.
    #
    # Let's try to find a 12-point configuration with 6 ordinary lines via a small search
    # seeded by the best grid, but using a smarter move: swapping points or moving to
    # "hot" spots (intersections of existing lines).
    
    best_points = [p[:] for p in grid_3x4]
    best_count = c1
    if c2 < best_count:
        best_points = [p[:] for p in grid_4x3]
        best_count = c2
        
    if best_count <= 6:
        return best_points

    # Local search with "line intersection" moves
    # Instead of moving to nearby integer coords, move to intersections of lines defined by other pairs.
    # This is a "constructive" local search.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_x, old_y = best_points[i]
            
            # Generate candidate new positions
            # 1. Nearby integers
            candidates = set()
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0: continue
                    candidates.add((old_x + dx, old_y + dy))
            
            # 2. Intersections of lines formed by pairs of other points
            # This is expensive to generate all, so we sample.
            # Actually, for n=12, generating all O(n^2) lines is fast.
            # Lines from points excluding i:
            other_indices = [j for j in range(12) if j != i]
            lines = []
            for a, b in combinations(other_indices, 2):
                p1 = best_points[a]
                p2 = best_points[b]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0: continue
                # Line: dy*x - dx*y = c
                c = dy*p1[0] - dx*p1[1]
                lines.append((dx, dy, c))
            
            # Find intersections of these lines
            # To avoid O(L^2) intersections, we just check if moving i to a point on a line
            # that already has 2 points makes it 3.
            # This is equivalent to: for each line L with 2 points (from other_indices),
            # try placing i on L. But i must be integer.
            # So for each line (dx, dy, c) with count 2, we want integer (x,y) such that
            # dy*x - dx*y = c.
            # This is a Diophantine equation.
            
            # Simpler: Just do the grid search above, it's fast enough for 12 points.
            # The grid search above is 49 moves per point * 12 points * 144 ops = ~85k ops per pass.
            # 1000 passes = 85M ops. In Python, this might be too slow if not optimized.
            # Let's optimize the count function or limit passes.
            
            for new_x, new_y in candidates:
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                new_point = (new_x, new_y)
                is_distinct = True
                for j in range(12):
                    if j != i and (best_points[j][0], best_points[j][1]) == new_point:
                        is_distinct = False
                        break
                
                if not is_distinct:
                    continue
                
                best_points[i] = [new_x, new_y]
                new_count = count_ordinary_lines(best_points)
                
                if new_count < best_count:
                    best_count = new_count
                    improved = True
                else:
                    best_points[i] = [old_x, old_y]
                    
            if time.time() > deadline:
                break
        
        if best_count <= 6:
            break

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(count_ordinary_lines(pts))
