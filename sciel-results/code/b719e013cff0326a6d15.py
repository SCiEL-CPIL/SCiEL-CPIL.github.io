"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _score_points(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    lines = {}
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
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    exact_4 = 0
    has_5 = False
    for key, pts_on_line in lines.items():
        cnt = len(pts_on_line)
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

def _try_line_completion(points, target_idx, target_pt, rng):
    """Try to place a point to complete a line with 3 existing points."""
    n = 16
    # Find all lines with exactly 3 points from the 15 other points
    other = [i for i in range(n) if i != target_idx]
    lines_3 = {}
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            for k in range(j + 1, len(other)):
                p1, p2, p3 = points[other[i]], points[other[j]], points[other[k]]
                # Check if collinear
                dx1 = p2[0] - p1[0]
                dy1 = p2[1] - p1[1]
                dx2 = p3[0] - p1[0]
                dy2 = p3[1] - p1[1]
                if dx1 * dy2 - dy1 * dx2 != 0:
                    continue
                # They are collinear; find the line
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
    
    # For each line with 3 points, try to find a 4th point on that line
    candidates = []
    for key, pts_on_line in lines_3.items():
        a, b, c = key
        # The line is a*x + b*y = c
        # We need to find an integer point (x,y) on this line that's not already one of the 3
        # and not equal to any existing point
        # Try a few positions along the line
        # Parametrize: if b != 0, y = (c - a*x)/b; if b == 0, x = c/a
        # Let's try a few x values near the existing points
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
                            # Check not equal to any existing point
                            if not any(pt == points[i] for i in range(n)):
                                candidates.append(pt)
            else:
                # b == 0, so a*x = c, x = c/a
                if a != 0 and c % a == 0:
                    x_val = c // a
                    if x_val == x_try and abs(x_val) <= 1000:
                        # Any y works, but we need it to be on the line
                        # Actually if b=0, the line is x = c/a (vertical)
                        # So any y is valid
                        for y_try in [points[list(pts_on_line)[0]][1] - 1, points[list(pts_on_line)[0]][1] + 1]:
                            if abs(y_try) <= 1000:
                                pt = [x_try, y_try]
                                if pt != target_pt:
                                    if not any(pt == points[i] for i in range(n)):
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
    
    # Try multiple starting configurations
    starting_configs = [
        [[x, y] for x in range(4) for y in range(4)],  # 4x4 grid
        [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 0] + 
         [[x, y] for x in range(4) for y in range(4) if (x + y) % 2 == 1][:8],  # checkerboard-ish
    ]
    
    for restart in range(80):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            current = [p[:] for p in base]
        elif restart == 1:
            current = [p[:] for p in starting_configs[1]]
        elif rng.random() < 0.5:
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
        max_no_improve = 800
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            idx = rng.randint(0, 15)
            old = current[idx][:]
            
            # Sometimes try line completion
            if rng.random() < 0.3:
                new_pt = _try_line_completion(current, idx, old, rng)
                if new_pt is None:
                    new_x = old[0] + rng.randint(-5, 5)
                    new_y = old[1] + rng.randint(-5, 5)
                    new_pt = [new_x, new_y]
            else:
                new_x = old[0] + rng.randint(-5, 5)
                new_y = old[1] + rng.randint(-5, 5)
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
