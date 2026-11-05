import random
import time
from itertools import combinations

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line to set of points on that line
    # Line represented by normalized (A, B, C) for Ax + By + C = 0
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0:
                pass
            else:
                # Simple normalization: divide by gcd if integers
                # But points are integers, so A,B,C are integers.
                # We just need a unique key.
                pass
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    best_score = float('inf')
    best_points = None
    
    # 1. Try some predefined configurations
    candidates = []
    
    # 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_3x4)
    
    # 4x3 Grid
    grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_4x3)
    
    # 2 lines of 6 (intersecting at 0,0)
    # L1: y=0, x=1..6
    # L2: x=0, y=1..6
    # Plus (0,0) is shared? No, distinct points.
    # L1: (1,0)..(6,0)
    # L2: (0,1)..(0,6)
    # Total 12.
    two_lines = [(i, 0) for i in range(1, 7)] + [(0, i) for i in range(1, 7)]
    candidates.append(two_lines)
    
    # 3 lines of 4 (concurrent)
    # L1: (1,0)..(4,0)
    # L2: (0,1)..(0,4)
    # L3: (1,1)..(4,4) ? No, (1,1) is not on y=x if origin is (0,0).
    # Let's use y=x. (1,1)..(4,4).
    # Intersections:
    # L1 (y=0) and L3 (y=x) intersect at (0,0).
    # L2 (x=0) and L3 (y=x) intersect at (0,0).
    # So (0,0) is the common point.
    # Points:
    # (0,0)
    # L1: (1,0), (2,0), (3,0), (4,0)
    # L2: (0,1), (0,2), (0,3), (0,4)
    # L3: (1,1), (2,2), (3,3), (4,4)
    # Total 1 + 4 + 4 + 4 = 13 points. Too many.
    # Remove one from each?
    # L1: (1,0), (2,0)
    # L2: (0,1), (0,2)
    # L3: (1,1), (2,2)
    # (0,0)
    # Total 7.
    
    # Let's just evaluate the candidates
    for pts in candidates:
        if len(pts) != 12: continue
        if len(set(map(tuple, pts))) != 12: continue
        # Check bounds
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts): continue
        
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
        if time.time() > deadline:
            break
            
    # 2. Local Search
    # Start from the best candidate found so far
    if best_points is None:
        best_points = grid_3x4
        best_score = count_ordinary_lines(best_points)
        
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Simple random perturbation
    # Pick a random point, move it to a random new location
    # If score improves, keep it.
    
    max_iterations = 10000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random point
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Generate new point
        # Try to keep it somewhat close or random in a small range
        # To find new collinearities, random in a larger box is better
        new_x = random.randint(-10, 10)
        new_y = random.randint(-10, 10)
        
        # Ensure distinct
        if (new_x, new_y) == (old_x, old_y):
            continue
        if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
            continue
            
        current_points[idx] = [new_x, new_y]
        
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            
    return best_points

# EVOLVE-BLOCK-START
def construct():
    # Use the logic from the scratch design
    import random
    import time
    from itertools import combinations

    def count_ordinary_lines(points):
        n = len(points)
        if n < 3:
            return 0
        
        lines = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                
                if A == 0 and B == 0:
                    continue
                
                # Normalize to handle -0 and signs
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
                
        ordinary_count = 0
        for pts in lines.values():
            if len(pts) == 2:
                ordinary_count += 1
                
        return ordinary_count

    start_time = time.time()
    deadline = start_time + 110 

    best_score = float('inf')
    best_points = None

    candidates = []
    
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_3x4)
    
    two_lines = [(i, 0) for i in range(1, 7)] + [(0, i) for i in range(1, 7)]
    candidates.append(two_lines)
    
    # Try a 3-3-3-3 configuration on 4 concurrent lines?
    # L1: (1,0)..(3,0)
    # L2: (0,1)..(0,3)
    # L3: (1,1)..(3,3)
    # L4: (-1,1)..(-3,3)? No.
    
    for pts in candidates:
        if len(pts) != 12: continue
        if len(set(map(tuple, pts))) != 12: continue
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts): continue
        
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
        if time.time() > deadline:
            break
            
    if best_points is None:
        best_points = grid_3x4
        best_score = count_ordinary_lines(best_points)
        
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        new_x = random.randint(-20, 20)
        new_y = random.randint(-20, 20)
        
        if (new_x, new_y) == (old_x, old_y):
            continue
        if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
            continue
            
        current_points[idx] = [new_x, new_y]
        
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
            
    return best_points
# EVOLVE-BLOCK-END
