import time
import random
from math import gcd

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    Returns (count, line_map) where line_map maps line keys to sets of point indices.
    """
    n = len(points)
    if n < 3:
        return 0, {}
    
    line_map = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            g = _gcd(_gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
                
            # Canonicalize the line representation
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    ordinary = 0
    for key, idxs in line_map.items():
        if len(idxs) == 2:
            ordinary += 1
            
    return ordinary, line_map

def _generate_candidates(points, line_map, existing_points_set):
    """
    Generate candidate positions for moving points.
    Focuses on:
    1. Extending lines with 2 or 3 points.
    2. Intersecting pairs of lines (to create points on multiple lines).
    """
    candidates = set()
    
    # 1. Extend lines
    for key, idxs in line_map.items():
        if len(idxs) < 2:
            continue
        A, B, C = key
        # The line is Ax + By + C = 0
        # We need integer points on this line.
        # Find one point on the line.
        # If B != 0, x = 0 => By = -C => y = -C/B (if B divides C)
        # If B == 0, Ax + C = 0 => x = -C/A (if A divides C)
        
        # A more robust way: find two points on the line and extend.
        pts_on_line = [points[i] for i in idxs]
        if len(pts_on_line) >= 2:
            p1 = pts_on_line[0]
            p2 = pts_on_line[1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Reduce direction vector
            g = _gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Generate points along this line
            for k in range(-10, 11):
                if k == 0 or k == 1:
                    continue # Skip existing points
                px = p1[0] + k * dx
                py = p1[1] + k * dy
                if abs(px) <= 10**9 and abs(py) <= 10**9:
                    candidates.add((px, py))
            
            # If there's a third point, it might define a different step? 
            # No, if 3 points are collinear, they are on the same line.
            # But if we have 3 points, we might want to extend from the ends.
            if len(pts_on_line) == 3:
                p3 = pts_on_line[2]
                # Check if p3 is on the same "step"
                # Just extending from p1 and p2 is usually enough if we use small k.
                # Let's also try extending from p3
                for k in range(-10, 11):
                    if k == 0 or k == 1:
                        continue
                    px = p3[0] + k * dx
                    py = p3[1] + k * dy
                    if abs(px) <= 10**9 and abs(py) <= 10**9:
                        candidates.add((px, py))

    # 2. Intersect pairs of lines
    # This is more expensive but very powerful.
    # We only do this for lines with size >= 2.
    # To keep it fast, we limit the number of intersections checked.
    line_keys = list(line_map.keys())
    random.shuffle(line_keys)
    
    checked = 0
    max_checks = 200 # Limit intersections to avoid explosion
    
    for i in range(len(line_keys)):
        if checked > max_checks:
            break
        k1 = line_keys[i]
        A1, B1, C1 = k1
        for j in range(i + 1, len(line_keys)):
            if checked > max_checks:
                break
            k2 = line_keys[j]
            A2, B2, C2 = k2
            checked += 1
            
            # Solve:
            # A1 x + B1 y = -C1
            # A2 x + B2 y = -C2
            # Determinant
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel or same line
            
            # x = (B1*(-C2) - B2*(-C1)) / det = (B1*C1 - B2*C2) / det
            # y = (A2*(-C1) - A1*(-C2)) / det = (A1*C2 - A2*C1) / det
            
            num_x = B1 * C1 - B2 * C2
            num_y = A1 * C2 - A2 * C1
            
            if num_x % det != 0 or num_y % det != 0:
                continue # Not integer solution
                
            px = num_x // det
            py = num_y // det
            
            if abs(px) <= 10**9 and abs(py) <= 10**9:
                candidates.add((px, py))
                
    # 3. Local perturbations (fallback)
    for p in points:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                px, py = p[0] + dx, p[1] + dy
                if abs(px) <= 10**9 and abs(py) <= 10**9:
                    candidates.add((px, py))
                    
    return list(candidates)

def _local_search(points, deadline_fn, max_iter=5000):
    """
    Local search using structural candidates.
    """
    current = [list(p) for p in points]
    current_score, current_line_map = _count_ordinary_lines(current)
    existing_set = set(map(tuple, current))
    
    for iteration in range(max_iter):
        if deadline_fn() > 115:
            break
            
        # Generate candidates
        cands = _generate_candidates(current, current_line_map, existing_set)
        random.shuffle(cands)
        
        improved = False
        best_new_score = current_score
        best_new_points = None
        best_new_idx = -1
        
        # Try moving each point to each candidate
        # Optimization: Don't evaluate all candidates for all points if we find a good one quickly?
        # No, we need to find the best move or any improving move.
        # Since N=12 and Cands might be ~100-500, 12*500 = 6000 evals per iteration.
        # Each eval is O(144) line calculations. ~1M ops per iteration.
        # 5000 iterations * 1M ops = 5B ops. Too slow.
        
        # We need to be smarter.
        # Instead of trying all candidates for all points, we can:
        # 1. Pick a random point to move.
        # 2. Try a subset of candidates for that point.
        
        # Let's try a random point and a random subset of candidates.
        idx = random.randint(0, 11)
        orig_pt = current[idx]
        
        # Limit candidates per point per iteration to keep it fast
        sample_cands = random.sample(cands, min(20, len(cands)))
        
        for cand in sample_cands:
            if cand == tuple(orig_pt):
                continue
            if cand in existing_set:
                # If cand is occupied by another point, we would need to swap or move that point too.
                # For simplicity, skip if occupied, unless we implement swaps.
                # Swaps are complex. Let's just skip occupied spots.
                # This might hinder progress if we need to swap.
                # But "moving to an empty spot" is usually sufficient for local search.
                continue
                
            new_pts = [list(p) for p in current]
            new_pts[idx] = list(cand)
            
            # Quick check: distinctness is guaranteed by existing_set check
            s, _ = _count_ordinary_lines(new_pts)
            
            if s < best_new_score:
                best_new_score = s
                best_new_points = new_pts
                best_new_idx = idx
                if s <= 6: # Theoretical minimum
                    improved = True
                    break # Found optimal, stop searching for better
            
        if best_new_points is not None:
            current = best_new_points
            current_score = best_new_score
            existing_set = set(map(tuple, current))
            current_line_map = _count_ordinary_lines(current)[1]
            improved = True
            if current_score <= 6:
                break # Can't do better than 6
        else:
            # No improvement found with this random point/cand selection
            # If we try many times without improvement, break
            if iteration > 100:
                # Try a few more random points before giving up
                pass
            # Actually, if no improvement, we might be stuck.
            # Let's just break if we've done a lot of iterations without improvement.
            # For now, just continue to next iteration.
            # To avoid infinite loop, we break if no improvement in last 50 iterations.
            # (Simplified: just let it run until deadline or max_iter)
            
    return current, current_score

def construct():
    start = time.time()
    deadline = lambda: time.time() - start
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    configs = []
    
    # 1. 3x4 Grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2. 4x3 Grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3. 3x3 Grid + 3 extensions (known good start)
    grid33 = [[x, y] for x in range(3) for y in range(3)]
    exts = [
        [[3, 0], [0, 3], [3, 3]],
        [[-1, 1], [1, -1], [4, 2]],
        [[2, 3], [3, 2], [4, 1]],
        [[1, 3], [3, 1], [4, 0]],
        [[3, 1], [1, 3], [3, 3]]
    ]
    for e in exts:
        configs.append(grid33 + e)
        
    # 4. Two parallel lines with 6 points each
    configs.append([[x, 0] for x in range(6)] + [[x, 2] for x in range(6)])
    
    # 5. Hexagonal-ish
    configs.append([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[1,2],[2,2],[0,2],[3,0],[3,1],[3,2]])
    
    # 6. Randomized starts
    for _ in range(10):
        pts = []
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        configs.append([list(p) for p in pts])
        
    # Evaluate and search from each config
    for cfg in configs:
        if deadline() > 110:
            break
            
        # Validate distinctness
        if len(set(map(tuple, cfg))) != 12:
            continue
            
        # Initial score
        s, _ = _count_ordinary_lines(cfg)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in cfg]
            
        # Local search
        if s <= best_score: # Only search if it's competitive
            p, s_new = _local_search(cfg, deadline, max_iter=2000)
            if s_new < best_score:
                best_score = s_new
                best_points = p
                if best_score <= 6:
                    break # Found optimal
    
    if best_points is None:
        best_points = [[x, y] for x in range(3) for y in range(4)]
        
    return best_points
