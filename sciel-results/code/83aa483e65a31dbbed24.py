import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key for two points."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _evaluate(points):
    """
    Returns (is_valid, score, line_counts).
    score: Number of lines containing exactly 4 points.
    line_counts: dict mapping line_key to count of points.
    """
    n = len(points)
    line_pair_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            key = _line_key(x1, y1, points[j][0], points[j][1])
            if key is not None:
                line_pair_counts[key] += 1
            
    valid = True
    score = 0
    line_counts = {}
    
    for key, count in line_pair_counts.items():
        k = -1
        # Solve k(k-1)/2 = count for k
        # k is small (<= 16)
        for possible_k in range(2, 17):
            if possible_k * (possible_k - 1) // 2 == count:
                k = possible_k
                break
        if k == -1:
            continue
        line_counts[key] = k
        if k >= 5:
            valid = False
            # We can break early if we just want validity, 
            # but we might want the score for invalid states? 
            # No, invalid states score 0.
            # However, for SA, we might want to know *how* invalid it is?
            # Usually, just returning 0 is enough.
            pass
        
    if not valid:
        return False, 0, line_counts
        
    for k in line_counts.values():
        if k == 4:
            score += 1
            
    return True, score, line_counts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def _sa_search(initial_points, max_time):
        current = [list(p) for p in initial_points]
        c_valid, c_score, _ = _evaluate(current)
        if not c_valid:
            c_score = -1000
            c_valid = True # Treat as very bad but allow moves
        
        best = [list(p) for p in current]
        best_score = c_score
        
        temp = 5.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            move_type = random.random()
            cand = [list(p) for p in current]
            
            if move_type < 0.4:
                # 1-point jitter
                idx = random.randint(0, 15)
                old_x, old_y = cand[idx]
                # Random small move
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                    continue
                cand[idx] = [new_x, new_y]
                
            elif move_type < 0.7:
                # 1-point snap to 3-point line
                # Find all lines with exactly 3 points
                _, _, line_counts = _evaluate(current)
                # Filter lines with k=3
                lines_3 = [k for k, v in line_counts.items() if v == 3]
                if lines_3:
                    line_key = random.choice(lines_3)
                    dx, dy, c = line_key
                    # Find a point not on this line to move
                    # First, identify which points are on this line
                    pts_on_line = set()
                    for i in range(16):
                        x1, y1 = current[i]
                        # Check if point is on line
                        # Line equation: dx*y - dy*x = c ? 
                        # Key was (dx, dy, c) where c = dy*x1 - dx*y1
                        # So dy*x - dx*y = c
                        if dy * x1 - dx * y1 == c:
                            pts_on_line.add(i)
                    
                    candidates_idx = [i for i in range(16) if i not in pts_on_line]
                    if candidates_idx:
                        idx = random.choice(candidates_idx)
                        # Find a new position on this line
                        # Pick a random existing point on the line as reference
                        ref_idx = random.choice(list(pts_on_line))
                        rx, ry = current[ref_idx]
                        # Step size
                        step = random.randint(1, 5)
                        sign = 1 if random.random() < 0.5 else -1
                        new_x = rx + dx * step * sign
                        new_y = ry + dy * step * sign
                        if abs(new_x) > 1000 or abs(new_y) > 1000:
                            continue
                        if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                            continue
                        cand[idx] = [new_x, new_y]
                    else:
                        continue
                else:
                    # Fallback to jitter
                    idx = random.randint(0, 15)
                    old_x, old_y = cand[idx]
                    new_x = old_x + random.randint(-5, 5)
                    new_y = old_y + random.randint(-5, 5)
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    if any(p[0] == new_x and p[1] == new_y for i, p in enumerate(cand) if i != idx):
                        continue
                    cand[idx] = [new_x, new_y]
                    
            else:
                # 2-point swap
                i1, i2 = random.sample(range(16), 2)
                cand[i1], cand[i2] = cand[i2], cand[i1]
                # Check distinctness (swapping distinct points keeps them distinct)
                
            cand_valid, cand_score, _ = _evaluate(cand)
            if not cand_valid:
                cand_score = -1000
            
            if cand_score > best_score:
                best = [list(p) for p in cand]
                best_score = cand_score
                no_improve = 0
            else:
                no_improve += 1
                
            # SA Acceptance
            if cand_score > c_score:
                current = cand
                c_score = cand_score
            else:
                delta = cand_score - c_score
                if delta < 0:
                    prob = pow(2.718281828, delta / max(temp, 0.01))
                    if random.random() < prob:
                        current = cand
                        c_score = cand_score
                        
            temp *= cooling
            
        return best, best_score

    # Initializations
    inits = []
    
    # 1. 4x4 Grid
    grid4 = [[x, y] for x in range(4) for y in range(4)]
    inits.append(grid4)
    
    # 2. Random
    random.seed(42)
    pts = set()
    while len(pts) < 16:
        x = random.randint(-10, 10)
        y = random.randint(-10, 10)
        pts.add((x, y))
    inits.append([list(p) for p in pts])
    
    # 3. Perturbed 4x4
    p_grid = [list(p) for p in grid4]
    for i in range(4):
        p_grid[i][0] += random.randint(-2, 2)
        p_grid[i][1] += random.randint(-2, 2)
    inits.append(p_grid)
    
    best_overall = None
    best_score_overall = -1
    
    # Run SA for each init
    for init in inits:
        if time.time() > deadline - 20:
            break
        alloc_time = min(deadline - time.time(), 40.0)
        res_pts, res_score = _sa_search(init, alloc_time)
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    # Final polish with remaining time
    if best_overall and time.time() < deadline - 5:
        res_pts, res_score = _sa_search(best_overall, deadline - time.time())
        if res_score > best_score_overall:
            best_score_overall = res_score
            best_overall = res_pts
            
    if best_overall is None:
        return grid4
        
    return best_overall
