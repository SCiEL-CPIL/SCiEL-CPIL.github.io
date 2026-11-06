"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _make_line_key(x1, y1, x2, y2):
    """Compute normalized line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * x1 + b * y1
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _make_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _score_points_fast(points):
    """Faster scoring using precomputed line keys for 16 points."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _make_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _get_lines_with_exactly_k(points, k):
    """Return list of (line_key, set_of_point_indices) for lines with exactly k points."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _make_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    result = []
    for key, pts in line_pts.items():
        if len(pts) == k:
            result.append((key, pts))
    return result

def _try_line_completion(points, target_idx, target_pt, rng):
    """Try to place a point to complete a line with 3 existing points."""
    n = 16
    other = [i for i in range(n) if i != target_idx]
    
    # Find all lines with exactly 3 points among the 'other' points
    lines_3 = {}
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            for k in range(j + 1, len(other)):
                oi, oj, ok = other[i], other[j], other[k]
                p1, p2, p3 = points[oi], points[oj], points[ok]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                if dx1 * dy2 - dy1 * dx2 != 0:
                    continue
                key = _make_line_key(p1[0], p1[1], p2[0], p2[1])
                if key is None:
                    continue
                if key not in lines_3:
                    lines_3[key] = set()
                lines_3[key].add(oi)
                lines_3[key].add(oj)
                lines_3[key].add(ok)
    
    candidates = []
    seen = set()
    for key, pts_on_line in lines_3.items():
        a, b, c = key
        # Try integer points on this line
        xs = sorted(set(points[p][0] for p in pts_on_line))
        for x_try in xs + [x - 2 for x in xs] + [x - 1 for x in xs] + [x + 1 for x in xs] + [x + 2 for x in xs]:
            if abs(x_try) > 1000:
                continue
            if b != 0:
                num = c - a * x_try
                if num % b == 0:
                    y_try = num // b
                    if abs(y_try) <= 1000:
                        pt = [x_try, y_try]
                        pt_tuple = tuple(pt)
                        if pt_tuple != tuple(target_pt) and pt_tuple not in seen:
                            if not any(pt == points[i] for i in range(n)):
                                seen.add(pt_tuple)
                                candidates.append(pt)
            else:
                # Horizontal line: y = c/a
                if a != 0 and c % a == 0:
                    y_val = c // a
                    if abs(y_val) <= 1000:
                        for x_try2 in xs + [x - 2 for x in xs] + [x - 1 for x in xs] + [x + 1 for x in xs] + [x + 2 for x in xs]:
                            if abs(x_try2) > 1000:
                                continue
                            pt = [x_try2, y_val]
                            pt_tuple = tuple(pt)
                            if pt_tuple != tuple(target_pt) and pt_tuple not in seen:
                                if not any(pt == points[i] for i in range(n)):
                                    seen.add(pt_tuple)
                                    candidates.append(pt)
    
    if candidates:
        return rng.choice(candidates)
    return None

def construct():
    deadline = time.time() + 145
    
    base = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_points(base)
    best_points = [p[:] for p in base]
    
    rng = random.Random(99)
    
    # Multiple diverse starting configurations
    starting_configs = [
        [[x, y] for x in range(4) for y in range(4)],
        [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 0] + 
         [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 1][:8],
        # Slightly perturbed grid
        [[x, y] for x in range(4) for y in range(4)],
        # Wider spread
        [[x*2, y*2] for x in range(4) for y in range(4)],
    ]
    
    for restart in range(100):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif restart == 1:
            current = [p[:] for p in starting_configs[1]]
        elif restart == 2:
            current = [p[:] for p in starting_configs[3]]
        elif rng.random() < 0.6:
            current = [p[:] for p in best_points]
        else:
            current = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(rng.randint(3, 12)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-5, 5)
                new_y = current[idx][1] + rng.randint(-5, 5)
                new_pt = [new_x, new_y]
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                if any(new_pt == p for p in current):
                    continue
                current[idx] = new_pt
        
        s, h5 = _score_points(current)
        if h5:
            continue
        current_score = s
        
        T = 0.5
        no_improve = 0
        max_no_improve = 1000
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            # Strategy: 40% line completion, 30% targeted 3-line move, 30% random
            r = rng.random()
            if r < 0.4:
                new_pt = _try_line_completion(current, idx, old, rng)
                if new_pt is None:
                    new_x = old[0] + rng.randint(-5, 5)
                    new_y = old[1] + rng.randint(-5, 5)
                    new_pt = [new_x, new_y]
            elif r < 0.7:
                # Targeted: find a line with exactly 3 points and try to move one of them
                # to create a new 4-point line while preserving the 3-point line
                lines_3 = _get_lines_with_exactly_k(current, 3)
                if lines_3:
                    _, pts_on_line = rng.choice(lines_3)
                    # Pick a point NOT on this line to move (to avoid breaking it)
                    candidates_idx = [i for i in range(16) if i not in pts_on_line]
                    if candidates_idx:
                        idx = rng.choice(candidates_idx)
                        old = current[idx][:]
                        new_x = old[0] + rng.randint(-4, 4)
                        new_y = old[1] + rng.randint(-4, 4)
                        new_pt = [new_x, new_y]
                    else:
                        new_x = old[0] + rng.randint(-5, 5)
                        new_y = old[1] + rng.randint(-5, 5)
                        new_pt = [new_x, new_y]
                else:
                    new_x = old[0] + rng.randint(-5, 5)
                    new_y = old[1] + rng.randint(-5, 5)
                    new_pt = [new_x, new_y]
            else:
                new_x = old[0] + rng.randint(-6, 6)
                new_y = old[1] + rng.randint(-6, 6)
                new_pt = [new_x, new_y]
            
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            if any(new_pt == p for p in current):
                continue
            
            current[idx] = new_pt
            s_new, h5_new = _score_points_fast(current)
            
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
            
            # Accept if better, or with probability based on temperature
            if s_new >= current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best_points = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx] = old
                no_improve += 1
            
            T = max(0.05, T * 0.998)
    
    # Final polish: try targeted moves on best_points
    if time.time() < deadline - 5:
        current = [p[:] for p in best_points]
        current_score = best_score
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            # Find lines with exactly 3 points
            lines_3 = _get_lines_with_exactly_k(current, 3)
            if not lines_3:
                break
            _, pts_on_line = rng.choice(lines_3)
            candidates_idx = [i for i in range(16) if i not in pts_on_line]
            if not candidates_idx:
                break
            idx = rng.choice(candidates_idx)
            old = current[idx][:]
            new_x = old[0] + rng.randint(-3, 3)
            new_y = old[1] + rng.randint(-3, 3)
            new_pt = [new_x, new_y]
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            if any(new_pt == p for p in current):
                continue
            current[idx] = new_pt
            s_new, h5_new = _score_points_fast(current)
            if not h5_new and s_new > current_score:
                current_score = s_new
                if s_new > best_score:
                    best_score = s_new
                    best_points = [p[:] for p in current]
            else:
                current[idx] = old
    
    return best_points
# EVOLVE-BLOCK-END
