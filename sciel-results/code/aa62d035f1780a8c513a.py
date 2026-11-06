import time
import random
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    """
    Returns a unique key for the line passing through p1 and p2.
    The line is represented as (A, B, C) for Ax + By = C, where gcd(A,B)=1.
    We normalize signs such that A > 0 or (A == 0 and B > 0).
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    
    if A == 0 and B == 0:
        return None
        
    g = gcd(abs(A), abs(B))
    if g > 0:
        A //= g
        B //= g
        C //= g
        
    # Normalize direction
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Also checks for validity (no 5 collinear).
    Returns (score, is_valid).
    """
    n = len(points)
    line_map = defaultdict(set)
    
    # Map each line to the set of point indices on it
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
                
    score = 0
    is_valid = True
    
    for key, indices in line_map.items():
        cnt = len(indices)
        if cnt >= 5:
            is_valid = False
            break
        if cnt == 4:
            score += 1
            
    return score, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_score = -1
    best_points = None
    
    rng = random.Random(42)
    
    # Strategy: Construct configurations using "parallel line bundles".
    # We define a direction (slope) and place points on parallel lines in that direction.
    # To get many lines with exactly 4 points, we want many lines to have exactly 4 points.
    # A 4x4 grid has 8 lines with 4 points (4 horizontal, 4 vertical).
    # Can we do better?
    # If we use a "staircase" or skewed grid, we might create more diagonals.
    # However, the 4x4 grid is quite dense.
    
    # Let's try to generate candidates that are "near-grid" but perturbed to create more 4-lines.
    # Or, we can try specific known configurations.
    # One known configuration for 16 points is the "affine plane" of order 4, but that has 5 points on lines (parallel classes of size 4 lines, each line has 4 points... wait.
    # Affine plane of order 4: 16 points. 5 parallel classes. Each class has 4 lines. Each line has 4 points.
    # Total lines with 4 points = 5 * 4 = 20.
    # This is a finite geometry structure. Can we embed it in integers?
    # The affine plane AG(2,4) can be represented over GF(4).
    # GF(4) = {0, 1, a, b} where a^2 = a+1.
    # Points (x, y) in GF(4)^2.
    # We need to map GF(4) to integers.
    # This is tricky because addition/multiplication in GF(4) don't map linearly to integers in a way that preserves collinearity for *all* lines unless we use a specific embedding.
    # However, we can just try to find an integer configuration that mimics this density.
    
    # Let's use a local search but with a much better initialization and move set.
    # Initialization: 4x4 grid is a strong local optimum.
    # Let's try to "break" the grid to create diagonal 4-lines.
    
    # Alternative: Random restarts with a specific "structured" random generator.
    # Generate points on a small grid, then shuffle coordinates.
    
    # Let's try a few specific structured patterns.
    
    # Pattern 1: 4x4 Grid
    pts_grid = [[x, y] for x in range(4) for y in range(4)]
    score, valid = _count_lines_exactly_4(pts_grid)
    if valid and score > best_score:
        best_score = score
        best_points = [list(p) for p in pts_grid]
        
    # Pattern 2: "Skewed" 4x4 Grid (Shear)
    # x' = x, y' = y + k*x
    # This preserves horizontal lines (4 points each).
    # Diagonals might change.
    for k in range(-2, 3):
        pts_shear = [[x, y + k*x] for x in range(4) for y in range(4)]
        # Check bounds
        if all(-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000 for p in pts_shear):
            score, valid = _count_lines_exactly_4(pts_shear)
            if valid and score > best_score:
                best_score = score
                best_points = [list(p) for p in pts_shear]

    # Pattern 3: Randomized Grid Perturbation
    # Start with 4x4 grid, move points slightly to create new 4-lines?
    # Actually, moving points *off* the grid usually breaks the existing 4-lines.
    # To gain, we must create *more* 4-lines than we break.
    # In a 4x4 grid, we have 8 lines of 4.
    # If we move one point, we break 3 lines (1 horiz, 1 vert, 1 diag? no, just the ones it was on).
    # A point in 4x4 grid is on 1 horiz, 1 vert. Maybe 1-2 diags.
    # If we move it, we lose those. We gain lines passing through the new position.
    # Unlikely to gain more than we lose locally.
    
    # Let's try a different structure: Two 4x2 grids? Or a 2x8 grid?
    # 2x8 grid: 2 horizontal lines of 8 (invalid, >4).
    
    # What about a "Checkerboard" subset? No, need 16 points.
    
    # Let's try a local search from the 4x4 grid, but allow "swap" moves instead of just "move".
    # Swapping two points might preserve the count of points on lines better?
    
    # Actually, the most promising lead is the AG(2,4) structure.
    # Let's try to find an integer representation of AG(2,4).
    # AG(2,4) has 20 lines of 4 points.
    # If we can find 16 integer points that form AG(2,4), we win big (score 20 vs 8).
    
    # How to construct AG(2,4) in integers?
    # GF(4) elements: 0, 1, w, w+1 (where w^2+w+1=0).
    # We can map 0->0, 1->1, w->2, w+1->3?
    # Addition table for GF(4):
    # 0+0=0, 0+1=1, 0+w=w, 0+w+1=w+1
    # 1+1=0, 1+w=w+1, 1+w+1=w
    # w+w=0, w+w+1=1
    # w+1+w+1=0
    # If we map to integers 0,1,2,3, does x+y mod 4 match GF(4) addition?
    # 1+1=2 in int, 0 in GF(4). No.
    # So simple modular arithmetic doesn't work.
    
    # However, we can use the fact that AG(2,4) is a set of 16 points.
    # Let's just run a heavy local search with multiple seeds and a larger neighborhood.
    
    def local_search(initial_points, max_iters=5000):
        current_points = [tuple(p) for p in initial_points]
        current_score, _ = _count_lines_exactly_4(current_points)
        best_local_score = current_score
        best_local_points = current_points[:]
        
        # Precompute line map for speed
        def get_line_map(pts):
            lm = defaultdict(set)
            for i in range(16):
                for j in range(i+1, 16):
                    key = _line_key(pts[i], pts[j])
                    if key:
                        lm[key].add(i)
                        lm[key].add(j)
            return lm
        
        line_map = get_line_map(current_points)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            # Try moving each point to a nearby location
            # To speed up, we only consider moves that keep points within a small box
            # around the current bounding box, or random jumps.
            
            # Pick a random point to move
            idx = rng.randint(0, 15)
            old_pos = current_points[idx]
            
            # Generate candidate new positions
            # 1. Nearby grid points
            candidates = set()
            # Explore a 21x21 box around current position
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx, ny = old_pos[0] + dx, old_pos[1] + dy
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        candidates.add((nx, ny))
                        
            # 2. Some random points
            for _ in range(20):
                x = rng.randint(-100, 100)
                y = rng.randint(-100, 100)
                candidates.add((x, y))
                
            # Shuffle
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            
            best_delta = -100
            best_move = None
            
            for cand in cand_list:
                if cand == old_pos:
                    continue
                
                # Check if cand is occupied
                if any(p == cand for p in current_points):
                    continue
                
                # Simulate move
                # Remove idx from all lines it's on
                # Add idx to all lines from new_pos to others
                
                # This is O(N) per candidate. 16 points.
                # Lines through old_pos:
                lines_old = set()
                for j in range(16):
                    if j == idx: continue
                    key = _line_key(old_pos, current_points[j])
                    if key: lines_old.add(key)
                    
                lines_new = set()
                for j in range(16):
                    if j == idx: continue
                    key = _line_key(cand, current_points[j])
                    if key: lines_new.add(key)
                    
                all_lines = lines_old.union(lines_new)
                
                delta = 0
                valid = True
                
                for key in all_lines:
                    old_size = len(line_map.get(key, set()))
                    in_old = key in lines_old
                    in_new = key in lines_new
                    
                    if in_old and in_new:
                        new_size = old_size
                    elif in_old:
                        new_size = old_size - 1
                    elif in_new:
                        new_size = old_size + 1
                    else:
                        new_size = old_size
                        
                    if new_size >= 5:
                        valid = False
                        break
                        
                    # Score change
                    if in_old and not in_new:
                        if old_size == 4: delta -= 1
                    elif in_new and not in_old:
                        if new_size == 4: delta += 1
                    # If both, size unchanged, score unchanged? 
                    # Wait, if size is 4 and stays 4, score is same.
                    # If size is 3 and becomes 4 (new), +1.
                    # If size is 4 and becomes 3 (old), -1.
                    # If size is 4 and stays 4 (both), 0.
                    
                if valid and delta > best_delta:
                    best_delta = delta
                    best_move = cand
                    
            if best_move is not None and best_delta > 0:
                # Apply move
                # Update line_map
                lines_old = set()
                for j in range(16):
                    if j == idx: continue
                    key = _line_key(old_pos, current_points[j])
                    if key: lines_old.add(key)
                    
                lines_new = set()
                for j in range(16):
                    if j == idx: continue
                    key = _line_key(best_move, current_points[j])
                    if key: lines_new.add(key)
                    
                all_lines = lines_old.union(lines_new)
                
                for key in all_lines:
                    if key in lines_old:
                        line_map[key].remove(idx)
                        if not line_map[key]:
                            del line_map[key]
                    if key in lines_new:
                        if key not in line_map:
                            line_map[key] = set()
                        line_map[key].add(idx)
                        
                current_points[idx] = best_move
                current_score += best_delta
                improved = True
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_points = current_points[:]
                    
            # If no improvement, we could try a random jump (simulated annealing),
            # but for now, let's just break and restart from a different seed/structure.
            # To avoid infinite loops, we limit iterations.
            if not improved:
                break
                
        return best_local_points, best_local_score

    # Run local search from 4x4 grid
    pts, score = local_search(pts_grid, max_iters=2000)
    if score > best_score:
        best_score = score
        best_points = [list(p) for p in pts]
        
    # Run local search from skewed grids
    for k in range(-3, 4):
        if time.time() > deadline:
            break
        pts_shear = [[x, y + k*x] for x in range(4) for y in range(4)]
        if all(-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000 for p in pts_shear):
            pts, score = local_search(pts_shear, max_iters=1000)
            if score > best_score:
                best_score = score
                best_points = [list(p) for p in pts]

    # Try a few random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        # Random points in a small box to encourage collisions
        pts_rand = [[rng.randint(-5, 5), rng.randint(-5, 5)] for _ in range(16)]
        # Ensure distinct
        unique_pts = []
        seen = set()
        for p in pts_rand:
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
        while len(unique_pts) < 16:
            p = [rng.randint(-5, 5), rng.randint(-5, 5)]
            if p not in seen:
                seen.add(p)
                unique_pts.append(p)
                
        pts, score = local_search(unique_pts, max_iters=1000)
        if score > best_score:
            best_score = score
            best_points = [list(p) for p in pts]

    if best_points is None:
        # Fallback to grid
        return [list(p) for p in pts_grid]
        
    return best_points

# Quick self-test / sanity check
if __name__ == "__main__":
    import sys
    # To avoid printing during evaluation, we just run construct
    # But for local testing:
    # res = construct()
    # print(res)
    # print(_count_lines_exactly_4(res))
    pass
