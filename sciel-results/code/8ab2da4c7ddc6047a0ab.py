import math
import random
import time
from collections import defaultdict
from math import gcd

def line_hash(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return ('v', x1)
    if y1 == y2:
        return ('h', y1)
    dy = y2 - y1
    dx = x2 - x1
    g = gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    elif dx_n == 0 and dy_n < 0:
        dy_n = -dy_n
    a = dy_n
    b = -dx_n
    c = dx_n * y1 - dy_n * x1
    g = gcd(abs(a), abs(b), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def evaluate(points):
    """Fast evaluation: count exact-4 lines, return -1 if invalid (5+ collinear or duplicates)."""
    n = len(points)
    if n != 16:
        return -1
    if len(set(map(tuple, points))) != 16:
        return -1
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = line_hash(points[i], points[j])
            line_counts[key] += 1
    score = 0
    for key, count in line_counts.items():
        disc = 1 + 8 * count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc != disc:
            return -1
        k = (1 + sqrt_disc) // 2
        if k >= 5:
            return -1
        if k == 4:
            score += 1
    return score

def get_line_points_on_segment(p1, p2, points_set, bounds=1000):
    """Given two points, find all integer points on the line through them within bounds,
    excluding those already in points_set. Returns list of candidate new positions."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return []
    step_x = dx // g
    step_y = dy // g
    
    candidates = []
    # Search backwards from p1
    k = 1
    while True:
        px = x1 - k * step_x
        py = y1 - k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        if (px, py) not in points_set:
            candidates.append((px, py))
        k += 1
        if k > 200:
            break
    
    # Search forwards from p1
    k = 1
    while True:
        px = x1 + k * step_x
        py = y1 + k * step_y
        if abs(px) > bounds or abs(py) > bounds:
            break
        if (px, py) not in points_set:
            candidates.append((px, py))
        k += 1
        if k > 200:
            break
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Generate diverse initial candidates
    initial_candidates = []
    
    # 4x4 grid variants with different scales and offsets
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            for ox in range(-3, 4):
                for oy in range(-3, 4):
                    pts = [[ox + sx*x, oy + sy*y] for x in range(4) for y in range(4)]
                    if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                        if len(set(tuple(p) for p in pts)) == 16:
                            initial_candidates.append(pts)
    
    # Small random perturbations of the base grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(20):
        pts = [p[:] for p in base_grid]
        for i in range(16):
            if random.random() < 0.4:
                pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
        if len(set(tuple(p) for p in pts)) == 16:
            initial_candidates.append(pts)
    
    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for pts in initial_candidates:
        s = evaluate(pts)
        if s >= 0:
            scored_candidates.append((s, pts))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_score = scored_candidates[0][0] if scored_candidates else 0
    best_points = [p[:] for p in scored_candidates[0][1]] if scored_candidates else base_grid
    
    def local_search(points, max_iters=30000, use_targeted=True):
        current_points = [p[:] for p in points]
        current_score = evaluate(current_points)
        if current_score < 0:
            return current_points, current_score
        
        best_local = [p[:] for p in current_points]
        best_local_score = current_score
        
        temp = 1.5
        cooling = 0.9995
        min_temp = 0.01
        no_improve = 0
        
        points_set = set(map(tuple, current_points))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Strategy: mix of targeted and random moves
            move_type = random.random()
            
            if use_targeted and move_type < 0.4:
                # Targeted: find a line with exactly 3 points, try to add a 4th
                # Find all lines and their point sets
                line_pts = defaultdict(set)
                for i in range(16):
                    for j in range(i + 1, 16):
                        key = line_hash(current_points[i], current_points[j])
                        line_pts[key].add(i)
                
                # Find lines with exactly 3 points
                three_lines = [(key, pts) for key, pts in line_pts.items() if len(pts) == 3]
                
                if three_lines:
                    key, pts = random.choice(three_lines)
                    # Pick two points on this line to define it
                    pts_list = list(pts)
                    p1_idx, p2_idx = random.sample(pts_list, 2)
                    p1 = current_points[p1_idx]
                    p2 = current_points[p2_idx]
                    
                    candidates = get_line_points_on_segment(p1, p2, points_set)
                    if candidates:
                        new_pt = random.choice(candidates)
                        idx = random.choice(pts_list)  # Move one of the existing points? No, we need to add a new point
                        # Actually, we can't add points. We need to MOVE a point to a new position.
                        # So: pick a point NOT on this line, and move it to a candidate position
                        non_line_pts = [i for i in range(16) if i not in pts]
                        if non_line_pts:
                            idx = random.choice(non_line_pts)
                            old_pt = current_points[idx][:]
                            new_pt_list = list(new_pt)
                            if new_pt_list in [current_points[i] for i in range(16) if i != idx]:
                                current_points[idx] = old_pt
                                no_improve += 1
                                continue
                            
                            current_points[idx] = new_pt_list
                            new_score = evaluate(current_points)
                            
                            if new_score == -1:
                                current_points[idx] = old_pt
                                no_improve += 1
                                continue
                            
                            delta = new_score - current_score
                            if delta > 0:
                                current_score = new_score
                                no_improve = 0
                                points_set = set(map(tuple, current_points))
                                if current_score > best_local_score:
                                    best_local_score = current_score
                                    best_local = [p[:] for p in current_points]
                            else:
                                if delta == 0:
                                    current_score = new_score
                                    points_set = set(map(tuple, current_points))
                                    no_improve += 1
                                else:
                                    prob = math.exp(delta / temp)
                                    if random.random() < prob:
                                        current_score = new_score
                                        points_set = set(map(tuple, current_points))
                                    else:
                                        current_points[idx] = old_pt
                                    no_improve += 1
                            temp = max(min_temp, temp * cooling)
                            if no_improve > 3000:
                                break
                            continue
            
            # Random move
            idx = random.randint(0, 15)
            old_pt = current_points[idx][:]
            
            # Random step size
            if random.random() < 0.7:
                step = random.randint(1, 4)
            else:
                step = random.randint(4, 15)
                
            new_x = old_pt[0] + random.randint(-step, step)
            new_y = old_pt[1] + random.randint(-step, step)
            
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
                
            new_pt = [new_x, new_y]
            
            is_distinct = True
            for i, p in enumerate(current_points):
                if i != idx and p == new_pt:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            current_points[idx] = new_pt
            new_score = evaluate(current_points)
            
            if new_score == -1:
                current_points[idx] = old_pt
                no_improve += 1
                continue
            
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                no_improve = 0
                points_set = set(map(tuple, current_points))
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = [p[:] for p in current_points]
            else:
                if delta == 0:
                    current_score = new_score
                    points_set = set(map(tuple, current_points))
                    no_improve += 1
                else:
                    prob = math.exp(delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                        points_set = set(map(tuple, current_points))
                    else:
                        current_points[idx] = old_pt
                    no_improve += 1
            
            temp = max(min_temp, temp * cooling)
            if no_improve > 3000:
                break
                
        return best_local, best_local_score
    
    # Run local search from top candidates
    for s, pts in scored_candidates[:10]:
        if time.time() > deadline:
            break
        result_pts, result_score = local_search(pts, max_iters=20000)
        if result_score > best_score:
            best_score = result_score
            best_points = [p[:] for p in result_pts]
    
    # Additional random restarts from best
    for _ in range(10):
        if time.time() > deadline:
            break
        pts = [p[:] for p in best_points]
        for i in range(16):
            if random.random() < 0.3:
                pts[i] = [pts[i][0] + random.randint(-5, 5), pts[i][1] + random.randint(-5, 5)]
        s = evaluate(pts)
        if s < 0:
            continue
        result_pts, result_score = local_search(pts, max_iters=10000)
        if result_score > best_score:
            best_score = result_score
            best_points = [p[:] for p in result_pts]
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s = evaluate(final)
    assert s >= 0, f"Invalid result: {s}"
    
    return final
