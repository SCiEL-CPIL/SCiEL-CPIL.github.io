"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
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
            key = (a, b, c)
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
    """Faster scoring using integer line keys and sorted point lists."""
    n = 16
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
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
            key = (a, b, c)
            if key not in line_pts:
                line_pts[key] = [i, j]
            else:
                lst = line_pts[key]
                if i not in lst and j not in lst:
                    lst.append(i)
                    lst.append(j)
                elif i not in lst:
                    lst.append(i)
                elif j not in lst:
                    lst.append(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts in line_pts.items():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _try_line_completion(points, target_idx, target_pt, rng):
    """Try to place a point to complete a line with 3 existing points."""
    n = 16
    other = [i for i in range(n) if i != target_idx]
    lines_3 = {}
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            for k in range(j + 1, len(other)):
                p1, p2, p3 = points[other[i]], points[other[j]], points[other[k]]
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                if dx1 * dy2 - dy1 * dx2 != 0:
                    continue
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n, dy_n = dx // g, dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n, dy_n = -dx_n, -dy_n
                a, b = dy_n, -dx_n
                c = a * p1[0] + b * p1[1]
                g2 = math.gcd(abs(a), abs(b))
                if g2 > 1:
                    a, b, c = a // g2, b // g2, c // g2
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                if key not in lines_3:
                    lines_3[key] = set()
                lines_3[key].add(other[i])
                lines_3[key].add(other[j])
                lines_3[key].add(other[k])
    
    candidates = []
    for key, pts_on_line in lines_3.items():
        a, b, c = key
        xs = set()
        for p in pts_on_line:
            xs.add(points[p][0])
        for x_try in sorted(xs) + [x - 1 for x in sorted(xs)] + [x + 1 for x in sorted(xs)]:
            if abs(x_try) > 1000:
                continue
            if b != 0:
                num = c - a * x_try
                if num % b == 0:
                    y_try = num // b
                    if abs(y_try) <= 1000:
                        pt = [x_try, y_try]
                        if pt != target_pt:
                            if not any(pt == points[i] for i in range(n)):
                                candidates.append(pt)
            else:
                if a != 0 and c % a == 0:
                    x_val = c // a
                    if x_val == x_try and abs(x_val) <= 1000:
                        for y_try in [points[list(pts_on_line)[0]][1] - 1, points[list(pts_on_line)[0]][1] + 1]:
                            if abs(y_try) <= 1000:
                                pt = [x_try, y_try]
                                if pt != target_pt:
                                    if not any(pt == points[i] for i in range(n)):
                                        candidates.append(pt)
    
    if candidates:
        return rng.choice(candidates)
    return None

def _find_3point_lines(points):
    """Find all lines with exactly 3 points from the 16."""
    n = 16
    line_pts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
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
            key = (a, b, c)
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    
    result = []
    for key, pts in line_pts.items():
        if len(pts) == 3:
            result.append((key, pts))
    return result

def construct():
    deadline = time.time() + 145
    
    base = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_points(base)
    best_points = [p[:] for p in base]
    
    rng = random.Random(99)
    
    starting_configs = [
        [[x, y] for x in range(4) for y in range(4)],
        [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 0] + 
         [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 1][:8],
    ]
    
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif restart == 1:
            current = [p[:] for p in starting_configs[1]]
        elif rng.random() < 0.4:
            current = [p[:] for p in best_points]
        else:
            current = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(rng.randint(2, 10)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-4, 4)
                new_y = current[idx][1] + rng.randint(-4, 4)
                new_pt = [new_x, new_y]
                if any(new_pt == p for p in current):
                    continue
                current[idx] = new_pt
        
        s, h5 = _score_points(current)
        if h5:
            continue
        current_score = s
        
        T = 0.5
        no_improve = 0
        max_no_improve = 600
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            # Choose mutation strategy
            strategy = rng.random()
            
            if strategy < 0.35:
                # Try to complete a 3-point line to make a 4-point line
                lines3 = _find_3point_lines(current)
                if lines3:
                    key, pts3 = rng.choice(lines3)
                    a, b, c = key
                    # Find a point on this line that's not already in pts3
                    # Try nearby integer points on the line
                    found = False
                    for pt_idx in list(pts3):
                        px, py = current[pt_idx]
                        # Try moving this point slightly along the line
                        for offset in [-3, -2, -1, 1, 2, 3]:
                            if b != 0:
                                # Line: a*x + b*y = c
                                # Try varying x
                                nx = px + offset
                                if abs(nx) > 1000:
                                    continue
                                num = c - a * nx
                                if b != 0 and num % b == 0:
                                    ny = num // b
                                    if abs(ny) <= 1000:
                                        new_pt = [nx, ny]
                                        if new_pt != current[pt_idx] and not any(new_pt == p for p in current):
                                            old = current[pt_idx][:]
                                            current[pt_idx] = new_pt
                                            s_new, h5_new = _score_points_fast(current)
                                            if not h5_new:
                                                if s_new > current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                                                    if s_new > current_score:
                                                        no_improve = 0
                                                        if s_new > best_score:
                                                            best_score = s_new
                                                            best_points = [p[:] for p in current]
                                                    else:
                                                        no_improve += 1
                                                    current_score = s_new
                                                    found = True
                                                    break
                                                else:
                                                    current[pt_idx] = old
                                                    no_improve += 1
                                            else:
                                                current[pt_idx] = old
                                                no_improve += 1
                            else:
                                # Vertical line: a*x = c
                                if a != 0:
                                    nx = c // a if c % a == 0 else px
                                    ny = py + offset
                                    if abs(nx) <= 1000 and abs(ny) <= 1000:
                                        new_pt = [nx, ny]
                                        if new_pt != current[pt_idx] and not any(new_pt == p for p in current):
                                            old = current[pt_idx][:]
                                            current[pt_idx] = new_pt
                                            s_new, h5_new = _score_points_fast(current)
                                            if not h5_new:
                                                if s_new > current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
                                                    if s_new > current_score:
                                                        no_improve = 0
                                                        if s_new > best_score:
                                                            best_score = s_new
                                                            best_points = [p[:] for p in current]
                                                    else:
                                                        no_improve += 1
                                                    current_score = s_new
                                                    found = True
                                                    break
                                                else:
                                                    current[pt_idx] = old
                                                    no_improve += 1
                                            else:
                                                current[pt_idx] = old
                                                no_improve += 1
                        if found:
                            break
                    T = max(0.05, T * 0.999)
                    continue
                else:
                    # No 3-point lines, fall through to random move
                    pass
            
            # Random perturbation
            idx = rng.randint(0, 15)
            old = current[idx][:]
            step = rng.randint(1, 6)
            new_x = old[0] + rng.choice([-step, step])
            new_y = old[1] + rng.choice([-step, step])
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
            
            T = max(0.05, T * 0.999)
    
    return best_points
# EVOLVE-BLOCK-END
