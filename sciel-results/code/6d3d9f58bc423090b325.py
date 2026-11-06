"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _line_key(p1, p2):
    """Canonical line key (a,b,c) for line through p1, p2: a*x+b*y=c, gcd(a,b)=1, a>0 or (a==0,b>0)."""
    x1, y1 = p1
    x2, y2 = p2
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

def _score(points):
    """Return (num_exact_4_lines, has_5_collinear)."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _score_fast(points):
    """Same as _score but slightly optimized."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n, dy_n = dx // g, dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n = -dx_n, -dy_n
            a, b = dy_n, -dx_n
            c = a * xi + b * yi
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
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _try_complete_line(points, line_key, existing_indices, rng):
    """
    Given a line key and 3 existing points on it, try to find a 4th integer point
    on that line that doesn't conflict with existing points and stays in bounds.
    """
    a, b, c = line_key
    existing_set = set(tuple(points[i]) for i in range(16))
    # Parametrize the line
    if b != 0:
        # a*x + b*y = c => y = (c - a*x) / b
        # Find x range where |y| <= 1000 and |x| <= 1000
        # Try x values near existing points
        xs_near = set()
        for idx in existing_indices:
            xs_near.add(points[idx][0])
        candidate_xs = set()
        for x0 in xs_near:
            for dx in range(-5, 6):
                candidate_xs.add(x0 + dx)
        # Also try a broader range
        for x0 in xs_near:
            for dx in range(-20, 21, 3):
                candidate_xs.add(x0 + dx)
        
        for x_try in sorted(candidate_xs):
            if abs(x_try) > 1000:
                continue
            num = c - a * x_try
            if num % b != 0:
                continue
            y_try = num // b
            if abs(y_try) > 1000:
                continue
            pt = (x_try, y_try)
            if pt in existing_set:
                continue
            return [x_try, y_try]
    elif a != 0:
        # a*x = c => x = c/a (vertical line)
        if c % a != 0:
            return None
        x_val = c // a
        if abs(x_val) > 1000:
            return None
        ys_near = set()
        for idx in existing_indices:
            ys_near.add(points[idx][1])
        candidate_ys = set()
        for y0 in ys_near:
            for dy in range(-5, 6):
                candidate_ys.add(y0 + dy)
        for y_try in sorted(candidate_ys):
            if abs(y_try) > 1000:
                continue
            pt = (x_val, y_try)
            if pt in existing_set:
                continue
            return [x_val, y_try]
    return None

def _find_3_lines(points):
    """Find all lines with exactly 3 points from the given 16 points."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    result = {}
    for key, pts in line_pts.items():
        if len(pts) == 3:
            result[key] = pts
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = random.Random(42)
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score(best_points)
    
    # Strategy 1: Local search with point moves
    for restart in range(50):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            current = [p[:] for p in best_points]
        else:
            # Perturb best
            current = [p[:] for p in best_points]
            num_moves = rng.randint(2, 6)
            for _ in range(num_moves):
                idx = rng.randint(0, 15)
                current[idx] = [current[idx][0] + rng.randint(-5, 5),
                               current[idx][1] + rng.randint(-5, 5)]
        
        # Remove duplicates
        seen = set()
        unique = []
        for p in current:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) < 16:
            continue
        current = unique[:16]
        
        s, h5 = _score(current)
        if h5:
            continue
        
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in current]
        
        # Local search: try moving single points
        T = 0.3
        no_improve = 0
        max_iters = 3000
        while time.time() < deadline - 15 and no_improve < max_iters:
            idx = rng.randint(0, 15)
            old_pt = current[idx][:]
            new_pt = [old_pt[0] + rng.randint(-4, 4), old_pt[1] + rng.randint(-4, 4)]
            
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            if any(new_pt == p for p in current):
                continue
            
            current[idx] = new_pt
            s_new, h5_new = _score_fast(current)
            
            if h5_new:
                current[idx] = old_pt
                no_improve += 1
                continue
            
            if s_new >= s or rng.random() < math.exp((s_new - s) / T):
                s = s_new
                no_improve = 0
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
            else:
                current[idx] = old_pt
                no_improve += 1
            
            T *= 0.999
    
    # Strategy 2: Line-centric construction
    # Try to build configurations by explicitly choosing 4-point lines
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Start from a random perturbation of the grid
        current = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(rng.randint(3, 8)):
            idx = rng.randint(0, 15)
            current[idx] = [current[idx][0] + rng.randint(-6, 6),
                           current[idx][1] + rng.randint(-6, 6)]
        
        # Remove duplicates
        seen = set()
        unique = []
        for p in current:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                unique.append(p)
        if len(unique) < 16:
            continue
        current = unique[:16]
        
        s, h5 = _score(current)
        if h5:
            continue
        
        # Try to add 4th points to 3-point lines
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            
            three_lines = _find_3_lines(current)
            if not three_lines:
                break
            
            line_key, existing_pts = rng.choice(list(three_lines.items()))
            new_pt = _try_complete_line(current, line_key, existing_pts, rng)
            if new_pt is None:
                continue
            
            # Replace the worst point with the new one
            # Find the point that contributes least to 4-lines
            best_replace_idx = None
            best_gain = -1
            for idx in range(16):
                if current[idx] == new_pt:
                    continue
                old_pt = current[idx][:]
                current[idx] = new_pt
                s_test, h5_test = _score_fast(current)
                if not h5_test and s_test > s:
                    gain = s_test - s
                    if gain > best_gain:
                        best_gain = gain
                        best_replace_idx = idx
                current[idx] = old_pt
            
            if best_replace_idx is not None:
                current[best_replace_idx] = new_pt
                s += best_gain
                if s > best_score:
                    best_score = s
                    best_points = [p[:] for p in current]
            else:
                # Just do a random move
                idx = rng.randint(0, 15)
                current[idx] = [current[idx][0] + rng.randint(-3, 3),
                               current[idx][1] + rng.randint(-3, 3)]
                s_new, h5_new = _score_fast(current)
                if not h5_new and s_new >= s:
                    s = s_new
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in current]
                else:
                    # Revert
                    current[idx] = [best_points[idx][0] + rng.randint(-2, 2),
                                   best_points[idx][1] + rng.randint(-2, 2)]
                    s_new, h5_new = _score_fast(current)
                    if h5_new or s_new < s:
                        s = best_score
                        current = [p[:] for p in best_points]
    
    # Final verification
    final_score, final_h5 = _score(best_points)
    if final_h5:
        # Fallback to grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
