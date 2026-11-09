import random
import itertools
import time
import math

def is_collinear(p1, p2, p3):
    return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)

def has_four_collinear(points):
    # Check all combinations of 4
    for quad in itertools.combinations(points, 4):
        if is_collinear(quad[0], quad[1], quad[2]) and is_collinear(quad[0], quad[1], quad[3]):
            return True
    return False

def count_three_collinear(points):
    count = 0
    for tri in itertools.combinations(points, 3):
        if is_collinear(tri[0], tri[1], tri[2]):
            count += 1
    return count

def max_3_free_subset(points):
    n = len(points)
    coll = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear(points[i], points[j], points[k]):
                    coll[i][j] = coll[i][k] = coll[j][k] = True
                    
    best = 0
    def backtrack(start, current, size):
        nonlocal best
        if size > best:
            best = size
        if start >= n:
            return
        if size + (n - start) <= best:
            return
        valid = True
        for i in current:
            if coll[i][start]:
                valid = False
                break
        if valid:
            backtrack(start + 1, current + [start], size + 1)
        backtrack(start + 1, current, size)
        
    backtrack(0, [], 0)
    return best

def repair_4_collinear(pts, max_iters=1000):
    """
    Randomly move points to break any 4-collinearities.
    Returns True if successful, False otherwise.
    """
    for _ in range(max_iters):
        # Find first 4-collinear quad
        quad_found = None
        for quad in itertools.combinations(range(20), 4):
            if is_collinear(pts[quad[0]], pts[quad[1]], pts[quad[2]]) and is_collinear(pts[quad[0]], pts[quad[1]], pts[quad[3]]):
                quad_found = quad
                break
        
        if not quad_found:
            return True
        
        # Pick a random point from the quad to move
        idx = random.choice(quad_found)
        
        # Try to find a new random position that doesn't create new 4-collinearities
        # and doesn't collide with existing points
        moved = False
        for _ in range(100):
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if [new_x, new_y] in pts:
                continue
            
            old_p = pts[idx]
            pts[idx] = [new_x, new_y]
            
            if not has_four_collinear(pts):
                moved = True
                break
            else:
                pts[idx] = old_p
        
        if not moved:
            # If we failed to move it safely, try a wider random range or just force it
            # Forcing it might create new 4-collinearities, which the loop will handle in next iteration
            # But we need to ensure we don't loop forever.
            # Let's just pick a random distinct point and hope for the best, or restart
            # Actually, if we can't find a safe move, we might be stuck.
            # Let's try 1000 more times with wider range
            for _ in range(1000):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                if [new_x, new_y] in pts:
                    continue
                old_p = pts[idx]
                pts[idx] = [new_x, new_y]
                if not has_four_collinear(pts):
                    moved = True
                    break
                else:
                    pts[idx] = old_p
            
            if not moved:
                return False
    return False

def construct():
    random.seed(42)
    
    best_score = 20
    best_pts = None
    
    # Number of restarts
    num_restarts = 200
    
    for r in range(num_restarts):
        # 1. Initialize random points
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        
        # 2. Repair 4-collinearities
        if not repair_4_collinear(pts):
            continue
        
        # 3. Local Search to minimize max_3_free_subset
        current_score = max_3_free_subset(pts)
        
        if current_score < best_score:
            best_score = current_score
            best_pts = pts[:]
            
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                # Generate candidate moves
                candidates = []
                
                # Random moves
                for _ in range(10):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in [(p[0], p[1]) for p in pts if p != pts[i]]:
                        candidates.append([x, y])
                        
                # Moves on lines defined by other points
                for j in range(20):
                    if j == i: continue
                    for k in range(j+1, 20):
                        if k == i: continue
                        p2 = pts[j]
                        p3 = pts[k]
                        dx = p3[0] - p2[0]
                        dy = p3[1] - p2[1]
                        if dx == 0 and dy == 0: continue
                        
                        # Try a few points on this line
                        for t in [-2, -1, 2, 3]:
                            nx = p2[0] + t * dx
                            ny = p2[1] + t * dy
                            if -100 <= nx <= 100 and -100 <= ny <= 100:
                                if (nx, ny) not in [(p[0], p[1]) for p in pts if p != pts[i]]:
                                    candidates.append([nx, ny])
                
                # Evaluate candidates
                best_candidate = None
                best_candidate_score = current_score
                
                for cand in candidates:
                    old_p = pts[i]
                    pts[i] = cand
                    if not has_four_collinear(pts):
                        new_score = max_3_free_subset(pts)
                        if new_score < best_candidate_score:
                            best_candidate_score = new_score
                            best_candidate = cand
                    pts[i] = old_p
                
                if best_candidate is not None:
                    pts[i] = best_candidate
                    current_score = best_candidate_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = pts[:]
                    
                    # Early stop if we reach a very good score
                    if best_score <= 10:
                        break
            
            if best_score <= 10:
                break
        
        if best_score <= 10:
            break

    if best_pts is None:
        # Fallback: return a valid random set if we couldn't find a good one
        # This shouldn't happen
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        repair_4_collinear(pts)
        return pts
        
    return best_pts
