import math
import time
import random
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0) # Should not happen for distinct points
    dx //= g
    dy //= g
    # Normalize direction so that (dx, dy) is unique for the line regardless of point order
    # We want (dx, dy) to be the same for (A,B) and (B,A).
    # Standardize: make first non-zero component positive.
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # The line is defined by direction (dx, dy) and a point on it.
    # To make the key unique for the line, we can use the normalized direction and the "intercept" or a canonical point.
    # A robust key: (dx, dy, x1*dy - y1*dx) -- wait, this is the cross product, related to distance from origin.
    # Let's use the standard line equation Ax + By = C.
    # A = dy, B = -dx.
    # C = A*x1 + B*y1 = dy*x1 - dx*y1.
    # We need to normalize (A, B, C) such that gcd(A,B,C)=1 and sign is fixed.
    A = dy
    B = -dx
    C = A * x1 + B * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    # Fix sign: make the first non-zero among A, B, C positive.
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _score(pts):
    # Returns (score, is_valid)
    # score: number of lines with exactly 4 points
    # is_valid: False if any line has 5 or more points
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
            lines[key].add(j)
    
    score = 0
    for key, indices in lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False # Invalid
        elif cnt == 4:
            score += 1
    return score, True

def _generate_initial():
    # Start with a 4x4 grid, centered and scaled
    # Coordinates 0..3, shift to -1.5..1.5? No, keep integers.
    # Let's use a 4x4 grid from (0,0) to (3,3) scaled by 10 to allow room for movement.
    # Actually, small coordinates are better for finding integer intersections.
    # Let's try a 4x4 grid on small integers.
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x, y])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer
    
    best_pts = None
    best_score = -1
    
    # Initial candidate
    cur_pts = _generate_initial()
    cur_score, cur_valid = _score(cur_pts)
    
    if cur_valid and cur_score > best_score:
        best_score = cur_score
        best_pts = [list(p) for p in cur_pts]
    
    # Local Search
    # We will iterate through points and try to move them to nearby integer coordinates
    # or to intersections of existing lines.
    
    # To speed up, we'll precompute current lines
    # But lines change.
    
    # Strategy: 
    # 1. Get all 3-lines and 4-lines.
    # 2. For each point, consider moving it to:
    #    a. An intersection of two 3-lines (to potentially create a 4-line or maintain 4-line while breaking 5-line if any).
    #    b. A point on an existing 3-line extended (to create a 4-line).
    #    c. Random nearby points.
    
    # Since we start with a valid config (score 10, no 5s), we just want to increase score.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Compute lines
        lines = defaultdict(set)
        for i in range(16):
            for j in range(i + 1, 16):
                key = _line_key(cur_pts[i], cur_pts[j])
                lines[key].add(i)
                lines[key].add(j)
        
        # Identify 3-lines and 4-lines
        three_lines = {k: v for k, v in lines.items() if len(v) == 3}
        four_lines = {k: v for k, v in lines.items() if len(v) == 4}
        
        # Candidates for each point
        # We try to move each point to a new location that increases the score.
        
        for p_idx in range(16):
            if time.time() > deadline:
                break
                
            current_pos = tuple(cur_pts[p_idx])
            tried_positions = set()
            
            # Generate candidates
            candidates = set()
            
            # 1. Intersections of pairs of 3-lines not containing p_idx
            # This is O(N^2) per point, which is 16 * (approx 20^2) = 6400 ops. Acceptable.
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    k1 = tl_keys[i]
                    k2 = tl_keys[j]
                    # Get two points from each line to define the line equations
                    # Line is Ax+By=C
                    A1, B1, C1 = k1
                    A2, B2, C2 = k2
                    
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue # Parallel
                    # Intersection
                    x = (C1 * B2 - C2 * B1) / det
                    y = (A1 * C2 - A2 * C1) / det
                    # Check if integer
                    if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                        xi = int(round(x))
                        yi = int(round(y))
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            # 2. Points on 3-lines extended
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                # Get the line equation
                A, B, C = k
                # Pick two points from the line to determine direction
                idxs = list(v)
                p1 = cur_pts[idxs[0]]
                p2 = cur_pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                # Try steps from p1
                for t in range(-5, 6):
                    if t == 0: continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            # 3. Random nearby points
            for _ in range(5):
                xi = current_pos[0] + random.randint(-5, 5)
                yi = current_pos[1] + random.randint(-5, 5)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            # Evaluate candidates
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                if cand in tried_positions:
                    continue
                tried_positions.add(cand)
                
                # Temporarily move point
                old_pos = cur_pts[p_idx]
                cur_pts[p_idx] = list(cand)
                
                new_score, new_valid = _score(cur_pts)
                
                # Restore
                cur_pts[p_idx] = old_pos
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [list(p) for p in cur_pts]
    
    # Return the best valid configuration found
    if best_pts is None:
        # Fallback: return the initial grid if it was valid (it is)
        best_pts = _generate_initial()
        
    return best_pts
