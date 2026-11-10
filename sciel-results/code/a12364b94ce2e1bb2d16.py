import math
import time
import random
from collections import defaultdict, Counter
from itertools import combinations

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = A * x1 + B * y1
    g2 = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _points_on_line(A, B, C, limit=1000):
    """Find all integer points (x,y) with |x|,|y|<=limit on line Ax+By+C=0."""
    pts = []
    if A == 0 and B == 0:
        return pts
    if A == 0:
        # By + C = 0 => y = -C/B
        if C % B != 0:
            return pts
        y = -C // B
        if abs(y) > limit:
            return pts
        for x in range(-limit, limit + 1):
            pts.append((x, y))
    elif B == 0:
        # Ax + C = 0 => x = -C/A
        if C % A != 0:
            return pts
        x = -C // A
        if abs(x) > limit:
            return pts
        for y in range(-limit, limit + 1):
            pts.append((x, y))
    else:
        # Ax + By = -C
        g = math.gcd(abs(A), abs(B))
        if (-C) % g != 0:
            return pts
        A2 = A // g
        B2 = B // g
        C2 = -C // g
        # Find one solution
        # A2*x + B2*y = C2
        # Use extended gcd
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
        g2, x0, y0 = egcd(A2, B2)
        # General solution: x = x0*C2 + t*B2, y = y0*C2 - t*A2
        x_base = x0 * C2
        y_base = y0 * C2
        # Find t range
        # x = x_base + t*B2 in [-limit, limit]
        # y = y_base - t*A2 in [-limit, limit]
        t_min = -10**9
        t_max = 10**9
        if B2 > 0:
            t_min = max(t_min, math.ceil((-limit - x_base) / B2))
            t_max = min(t_max, math.floor((limit - x_base) / B2))
        elif B2 < 0:
            t_min = max(t_min, math.ceil((limit - x_base) / B2))
            t_max = min(t_max, math.floor((-limit - x_base) / B2))
        if A2 > 0:
            t_min = max(t_min, math.ceil((y_base - limit) / A2))
            t_max = min(t_max, math.floor((y_base + limit) / A2))
        elif A2 < 0:
            t_min = max(t_min, math.ceil((y_base + limit) / A2))
            t_max = min(t_max, math.floor((y_base - limit) / A2))
        t_min = max(t_min, -100000)
        t_max = min(t_max, 100000)
        for t in range(t_min, t_max + 1):
            x = x_base + t * B2
            y = y_base - t * A2
            if abs(x) <= limit and abs(y) <= limit:
                pts.append((x, y))
    return pts

def _score(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
    score = 0
    for key, indices in lines.items():
        cnt = len(indices)
        if cnt >= 5:
            return -1, False
        elif cnt == 4:
            score += 1
    return score, True

def _build_lines(pts):
    lines = defaultdict(set)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(pts[i], pts[j])
            lines[key].add(i)
    return lines

def _delta_score(pts, new_pos, p_idx, lines_dict):
    """Compute change in score when moving point p_idx to new_pos."""
    n = len(pts)
    old_lines = set()
    for key, idxs in lines_dict.items():
        if p_idx in idxs:
            old_lines.add(key)
    
    new_lines = defaultdict(set)
    for j in range(n):
        if j == p_idx:
            continue
        key = _line_key(new_pos, pts[j])
        new_lines[key].add(j)
    
    affected = old_lines | set(new_lines.keys())
    delta = 0
    valid = True
    
    for key in affected:
        old_full = len(lines_dict[key]) if key in lines_dict else 0
        new_full = len(new_lines[key]) + 1 if key in new_lines else 0
        
        old_contrib = 1 if old_full == 4 else 0
        new_contrib = 1 if new_full == 4 else 0
        
        if new_full >= 5:
            valid = False
        delta += new_contrib - old_contrib
    
    return delta, valid

def _generate_initial(seed=0):
    rng = random.Random(seed)
    pts = []
    for x in range(4):
        for y in range(4):
            px = x + rng.choice([0, 0, 0, 0, 1, -1])
            py = y + rng.choice([0, 0, 0, 0, 1, -1])
            pts.append([px, py])
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _generate_structured(seed=0):
    rng = random.Random(seed)
    pts = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(rng.randint(1, 3)):
        idx = rng.randint(0, 15)
        pts[idx][0] += rng.choice([-1, 1])
        pts[idx][1] += rng.choice([-1, 1])
    seen = set()
    unique_pts = []
    for p in pts:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append(p)
    while len(unique_pts) < 16:
        x = rng.randint(-3, 6)
        y = rng.randint(-3, 6)
        tp = (x, y)
        if tp not in seen:
            seen.add(tp)
            unique_pts.append([x, y])
    return unique_pts[:16]

def _local_search(cur_pts, cur_score, deadline, max_iters=50):
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        lines_dict = _build_lines(cur_pts)
        three_lines = {k: v for k, v in lines_dict.items() if len(v) == 3}
        four_lines = {k: v for k, v in lines_dict.items() if len(v) == 4}
        
        for p_idx in range(16):
            if time.time() > deadline:
                return cur_pts, cur_score, False
            
            current_pos = tuple(cur_pts[p_idx])
            candidates = set()
            
            tl_keys = [k for k, v in three_lines.items() if p_idx not in v]
            for i in range(len(tl_keys)):
                for j in range(i + 1, len(tl_keys)):
                    A1, B1, C1 = tl_keys[i]
                    A2, B2, C2 = tl_keys[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            for k, v in three_lines.items():
                if p_idx in v:
                    continue
                idxs = list(v)
                p1 = cur_pts[idxs[0]]
                p2 = cur_pts[idxs[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                for t in range(-4, 5):
                    if t == 0:
                        continue
                    xi = p1[0] + t * dx
                    yi = p1[1] + t * dy
                    if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                        candidates.add((xi, yi))
            
            all_lines = list(three_lines.keys()) + list(four_lines.keys())
            for i in range(len(all_lines)):
                for j in range(i + 1, len(all_lines)):
                    A1, B1, C1 = all_lines[i]
                    A2, B2, C2 = all_lines[j]
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x_num = C1 * B2 - C2 * B1
                    y_num = A1 * C2 - A2 * C1
                    if x_num % det == 0 and y_num % det == 0:
                        xi = x_num // det
                        yi = y_num // det
                        if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                            candidates.add((xi, yi))
            
            for _ in range(6):
                xi = current_pos[0] + random.randint(-4, 4)
                yi = current_pos[1] + random.randint(-4, 4)
                if -1000 <= xi <= 1000 and -1000 <= yi <= 1000:
                    candidates.add((xi, yi))
            
            best_new_pos = None
            best_new_score = cur_score
            current_set = set(map(tuple, cur_pts))
            
            for cand in candidates:
                if cand == current_pos or cand in current_set:
                    continue
                
                delta, new_valid = _delta_score(cur_pts, cand, p_idx, lines_dict)
                new_score = cur_score + delta
                
                if new_valid and new_score > best_new_score:
                    best_new_score = new_score
                    best_new_pos = cand
            
            if best_new_pos is not None:
                cur_pts[p_idx] = list(best_new_pos)
                cur_score = best_new_score
                improved = True
    
    return cur_pts, cur_score, improved

def _line_based_search(deadline, rng):
    """
    Search over sets of 4-point lines. Build a catalog of candidate lines,
    then greedily select a compatible set of lines, then place points.
    """
    # Generate candidate lines from a grid
    # We'll consider lines that pass through at least 3 grid points in a reasonable range
    candidates = set()
    
    # Horizontal, vertical, and diagonal lines through small grid
    for x in range(-3, 7):
        candidates.add((0, 1, -x))  # vertical x=const -> A=0,B=1,C=-x
    for y in range(-3, 7):
        candidates.add((1, 0, -y))  # horizontal y=const -> A=1,B=0,C=-y
    
    # Slopes 1, -1, 2, -2, 1/2, -1/2 through grid points
    for x0 in range(-2, 6):
        for y0 in range(-2, 6):
            for dx, dy in [(1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2)]:
                A = dy
                B = -dx
                C = A * x0 + B * y0
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                candidates.add((A, B, C))
    
    # Also add lines through pairs of grid points
    for x1 in range(-2, 6):
        for y1 in range(-2, 6):
            for x2 in range(-2, 6):
                for y2 in range(-2, 6):
                    if x1 == x2 and y1 == y2:
                        continue
                    A = y2 - y1
                    B = x1 - x2
                    C = A * x1 + B * y1
                    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                        A = -A
                        B = -B
                        C = -C
                    candidates.add((A, B, C))
    
    # Precompute points on each candidate line (within |x|,|y|<=1000, but we'll restrict to smaller)
    line_pts = {}
    for key in candidates:
        A, B, C = key
        pts = _points_on_line(A, B, C, limit=30)
        # Only keep lines with at least 4 integer points in a reasonable range
        if len(pts) >= 4:
            line_pts[key] = pts
    
    # Now greedily select lines
    # We want to select lines such that we can place 16 points on them
    # Each point can be on multiple lines
    # This is like a set cover / packing problem
    
    # Simpler approach: pick a random subset of lines, then find points that maximize coverage
    best_score = 0
    best_pts = None
    
    # Try multiple random line selections
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Randomly select 8-12 lines
        num_lines = rng.randint(8, 12)
        selected_lines = rng.sample(list(line_pts.keys()), min(num_lines, len(line_pts)))
        
        # Find points that appear on at least 2 selected lines (these are "junction" points)
        point_line_count = defaultdict(set)
        for li, key in enumerate(selected_lines):
            for p in line_pts[key]:
                point_line_count[p].add(li)
        
        # Sort points by number of lines they're on
        sorted_points = sorted(point_line_count.items(), key=lambda x: -len(x[1]))
        
        # Greedily select points
        pts = []
        used_lines = set()
        for p, line_ids in sorted_points:
            if len(pts) >= 16:
                break
            if len(line_ids) >= 1:
                pts.append(p)
                used_lines.update(line_ids)
        
        # If we don't have 16, add more points
        if len(pts) < 16:
            existing = set(pts)
            for key in selected_lines:
                for p in line_pts[key]:
                    if len(pts) >= 16:
                        break
                    if p not in existing:
                        pts.append(p)
                        existing.add(p)
        
        if len(pts) < 16:
            continue
        
        # Check validity and score
        score, valid = _score(pts)
        if valid and score > best_score:
            best_score = score
            best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def _improve_from_lines(start_pts, start_score, deadline):
    """Take a line-based solution and improve with local search."""
    pts = [list(p) for p in start_pts]
    score = start_score
    
    # Multiple rounds of local search
    for _ in range(3):
        if time.time() > deadline:
            break
        pts, score, _ = _local_search(pts, score, deadline, max_iters=20)
    
    return pts, score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score = -1
    
    # Strategy 1: Traditional local search from multiple starts
    for seed in range(10):
        if time.time() > deadline:
            break
        
        if seed % 2 == 0:
            cur_pts = _generate_initial(seed)
        else:
            cur_pts = _generate_structured(seed)
        
        cur_score, cur_valid = _score(cur_pts)
        
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [list(p) for p in cur_pts]
        
        if cur_valid and time.time() < deadline:
            cur_pts, cur_score, _ = _local_search(cur_pts, cur_score, deadline)
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
    
    # Strategy 2: Line-based search
    if time.time() < deadline:
        rng = random.Random(42)
        line_pts, line_score = _line_based_search(deadline, rng)
        if line_pts is not None and line_score > best_score:
            best_score = line_score
            best_pts = line_pts
        
        # Improve line-based solution
        if best_pts is not None and time.time() < deadline:
            improved_pts, improved_score = _improve_from_lines(best_pts, best_score, deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_pts = improved_pts
    
    # Final local search on best solution
    if best_pts is not None and time.time() < deadline:
        final_pts, final_score, _ = _local_search(best_pts, best_score, deadline, max_iters=30)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Fallback
    if best_pts is None or best_score < 10:
        pts = [[x, y] for x in range(4) for y in range(4)]
        best_pts = pts
        best_score, _ = _score(best_pts)
    
    return best_pts
