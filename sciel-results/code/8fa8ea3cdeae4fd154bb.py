import numpy as np
import random
import time

def construct():
    # 1. Solver for Max No-3-Collinear Subset
    # Optimized for speed using bitmasks and aggressive pruning.
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinear triples as bitmasks
        # triple_mask[i] is a bitmask of points collinear with i? No, let's list triples.
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
            
        # Convert triples to bitmasks for fast intersection
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        best = 0
        # Sort triples by something? Maybe just use the first active one.
        
        # We want to find the maximum subset S of {0..n-1} such that no triple is fully contained in S.
        # This is equivalent to finding the minimum vertex cover of the 3-uniform hypergraph of triples,
        # then answer = n - min_vertex_cover.
        # Or directly: Max Independent Set in 3-uniform hypergraph.
        
        def solve(candidates_mask, current_size):
            nonlocal best
            
            # Pruning 1: Upper bound
            if current_size + bin(candidates_mask).count('1') <= best:
                return
                
            # Find first triple fully contained in candidates
            # We can iterate through precomputed triples
            # Optimization: If we maintain a list of active triples, it's faster.
            # For N=20, iterating all triples (max ~1140) is fast enough if called sparingly.
            # But recursion depth can be high.
            
            # Better: Precompute for each mask? No, too big.
            # Let's find a triple to branch on.
            branch_triple_idx = -1
            for idx, mask in enumerate(triple_masks):
                if (mask & candidates_mask) == mask:
                    branch_triple_idx = idx
                    break
            
            if branch_triple_idx == -1:
                # No triples left in candidates, add all
                new_size = current_size + bin(candidates_mask).count('1')
                if new_size > best:
                    best = new_size
                return
            
            t = triples[branch_triple_idx]
            u, v, w = t
            
            # Branch 1: Exclude u
            solve(candidates_mask & ~(1 << u), current_size)
            # Branch 2: Exclude v
            solve(candidates_mask & ~(1 << v), current_size)
            # Branch 3: Exclude w
            solve(candidates_mask & ~(1 << w), current_size)
            
        solve((1 << n) - 1, 0)
        return best

    # 2. Validity Check
    def is_valid(points):
        if len(points) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if len(set(map(tuple, points))) != 20:
            return False
        
        n = len(points)
        # Check no 4 collinear
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    # 3. Constructive Heuristic: Line Packing
    # Goal: Create many collinear triples to reduce the max no-3-collinear subset.
    # Strategy: Select lines with small coefficients. Place 3 points on each line.
    # Ensure no line has 4 points.
    # Ensure points are distinct.
    
    random.seed(42)
    
    # Generate candidate lines
    # Lines ax + by = c
    # Small a, b to keep points within bounds [-100, 100]
    lines = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if a == 0 and b == 0:
                continue
            # Normalize
            if a < 0 or (a == 0 and b < 0):
                a, b = -a, -b
            # c range
            # Max |ax| + |by| <= 200. So c can be roughly -200 to 200.
            # But we want integer points.
            # Let's just pick random integer points and form lines?
            pass
            
    # Alternative: Start with a known good configuration and improve.
    # Known lower bound for n=20 is around 12-13?
    # Let's try to construct a configuration with many triples.
    # If we have 6 disjoint lines of 3 points, that's 18 points.
    # Plus 2 points not on any line?
    # Max no-3-collinear subset:
    # From each line of 3, we can pick at most 2.
    # So 6 * 2 = 12.
    # Plus the 2 free points = 14.
    # Can we do better?
    # If lines intersect, we might be able to pick fewer?
    # No, picking points on intersecting lines might force us to pick fewer from one line to avoid 3 on another?
    # Actually, if lines intersect, the intersection point is shared.
    
    # Let's try to generate a set of points that lies on a grid but perturbed to break 4-collinearities
    # while maintaining 3-collinearities.
    
    # Initialize with a 4x5 grid
    points = []
    for i in range(4):
        for j in range(5):
            points.append([i * 10, j * 10])
            
    # This grid has many 4-collinearities.
    # We need to break them.
    # Randomly move points to break 4-collinearities but try to keep 3-collinearities.
    
    valid = False
    attempts = 0
    while not valid and attempts < 2000:
        if is_valid(points):
            valid = True
            break
        
        # Find a 4-collinear set and move one point
        moved = False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                line_pts = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        line_pts.append(k)
                if len(line_pts) >= 2: # Total >= 4
                    # Move one of the points on this line
                    idx = random.choice(line_pts)
                    # Move to random nearby spot
                    new_x = points[idx][0] + random.randint(-5, 5)
                    new_y = points[idx][1] + random.randint(-5, 5)
                    if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                        collides = False
                        for l, p in enumerate(points):
                            if l != idx and p[0] == new_x and p[1] == new_y:
                                collides = True
                                break
                        if not collides:
                            points[idx] = [new_x, new_y]
                            moved = True
                            break
            if moved:
                break
        if not moved:
            # Fallback random move
            idx = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            collides = False
            for l, p in enumerate(points):
                if l != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if not collides:
                points[idx] = [new_x, new_y]
                moved = True
                
        if not moved:
            # Reset to grid if stuck
            points = []
            for i in range(4):
                for j in range(5):
                    points.append([i * 10, j * 10])
        
        attempts += 1
        
    if not valid:
        return [[x, x * x] for x in range(-10, 10)]

    current_score = max_no_3_collinear(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local Search
    time_limit = time.time() + 115
    
    while time.time() < time_limit:
        # Try to improve by moving a point
        idx = random.randint(0, 19)
        # Try several random moves
        improved = False
        for _ in range(10):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
            collides = False
            for i, p in enumerate(points):
                if i != idx and p[0] == new_x and p[1] == new_y:
                    collides = True
                    break
            if collides:
                continue
            
            old_p = points[idx][:]
            points[idx] = [new_x, new_y]
            
            if is_valid(points):
                new_score = max_no_3_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    improved = True
                    break
                else:
                    points[idx] = old_p
            else:
                points[idx] = old_p
                
        if not improved:
            # Try swapping two points
            i1, i2 = random.sample(range(20), 2)
            points[i1], points[i2] = points[i2][:], points[i1][:]
            if is_valid(points):
                new_score = max_no_3_collinear(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                else:
                    points[i1], points[i2] = points[i2][:], points[i1][:]
            else:
                points[i1], points[i2] = points[i2][:], points[i1][:]

    return best_points
