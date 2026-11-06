"""Bootstrap spawn: model-and-optimize for 16 integer points, maximize exact-4 lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
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
    return (a, b, c)

def _score_lines(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
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

def _local_search(points, deadline, rng):
    """Local search to improve the number of exact-4 lines."""
    current = [p[:] for p in points]
    current_score, h5 = _score_lines(current)
    if h5:
        return current, 0
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        idx = rng.randint(0, 15)
        old = current[idx][:]
        
        # Move strategies
        move_type = rng.random()
        if move_type < 0.5:
            # Random move with variable step size
            step = rng.randint(1, 20)
            new_x = old[0] + rng.randint(-step, step)
            new_y = old[1] + rng.randint(-step, step)
            new_pt = [new_x, new_y]
        elif move_type < 0.7:
            # Swap with another point
            jdx = rng.randint(0, 15)
            if jdx == idx:
                continue
            new_pt = current[jdx][:]
            # Swap
            current[idx], current[jdx] = current[jdx], current[idx]
            s_new, h5_new = _score_lines(current)
            if h5_new:
                current[idx], current[jdx] = current[jdx], current[idx]
                no_improve += 1
                continue
            if s_new > current_score or (T > T_min and rng.random() < math.exp((s_new - current_score) / T)):
                if s_new > current_score:
                    no_improve = 0
                    if s_new > best_score:
                        best_score = s_new
                        best = [p[:] for p in current]
                else:
                    no_improve += 1
                current_score = s_new
            else:
                current[idx], current[jdx] = current[jdx], current[idx]
                no_improve += 1
            T = max(T_min, T * cooling)
            continue
        else:
            # Try to complete a 3-point line
            # Find a line with 3 points
            line_pts = {}
            for i in range(16):
                for j in range(i + 1, 16):
                    key = _line_key(current[i], current[j])
                    if key is None: continue
                    if key not in line_pts:
                        line_pts[key] = set()
                    line_pts[key].add(i)
                    line_pts[key].add(j)
            
            three_lines = [pts for pts in line_pts.values() if len(pts) == 3]
            if three_lines:
                target_pts = rng.choice(three_lines)
                # Get line equation
                p1 = current[list(target_pts)[0]]
                p2 = current[list(target_pts)[1]]
                key = _line_key(p1, p2)
                a, b, c = key
                # Find a point on this line not in target_pts
                # Try to move a random point to this line
                # Pick a point not in target_pts
                candidates_idx = [i for i in range(16) if i not in target_pts]
                if candidates_idx:
                    move_idx = rng.choice(candidates_idx)
                    # Find integer point on line a*x + b*y = c
                    # a*x + b*y = c
                    # Try x values
                    found = False
                    for x_try in range(-1000, 1001, 1):
                        if abs(x_try) > 1000: continue
                        num = c - a * x_try
                        if b != 0:
                            if num % b == 0:
                                y_try = num // b
                                if abs(y_try) <= 1000:
                                    new_pt = [x_try, y_try]
                                    if new_pt != current[move_idx]:
                                        # Check if this creates 5-collinear
                                        # We are adding to a 3-line, so it becomes 4.
                                        # Check if it creates 5-collinear on other lines.
                                        # Simple check: just try it.
                                        old_pt = current[move_idx][:]
                                        current[move_idx] = new_pt
                                        s_new, h5_new = _score_lines(current)
                                        if h5_new:
                                            current[move_idx] = old_pt
                                            no_improve += 1
                                            continue
                                        if s_new > current_score or (T > T_min and rng.random() < math.exp((s_new - current_score) / T)):
                                            if s_new > current_score:
                                                no_improve = 0
                                                if s_new > best_score:
                                                    best_score = s_new
                                                    best = [p[:] for p in current]
                                            else:
                                                no_improve += 1
                                            current_score = s_new
                                        else:
                                            current[move_idx] = old_pt
                                            no_improve += 1
                                        found = True
                                        break
                    if found:
                        T = max(T_min, T * cooling)
                        continue
            # If no move made, just do a random move
            step = rng.randint(1, 10)
            new_x = old[0] + rng.randint(-step, step)
            new_y = old[1] + rng.randint(-step, step)
            new_pt = [new_x, new_y]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            continue
        if any(new_pt == p for p in current):
            continue
        
        current[idx] = new_pt
        s_new, h5_new = _score_lines(current)
        
        if h5_new:
            current[idx] = old
            no_improve += 1
            continue
        
        if s_new > current_score or (T > T_min and rng.random() < math.exp((s_new - current_score) / T)):
            if s_new > current_score:
                no_improve = 0
                if s_new > best_score:
                    best_score = s_new
                    best = [p[:] for p in current]
            else:
                no_improve += 1
            current_score = s_new
        else:
            current[idx] = old
            no_improve += 1
        
        T = max(T_min, T * cooling)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    rng = random.Random(42)
    
    # Start with a good base: 4x4 grid
    base = [[x, y] for x in range(4) for y in range(4)]
    base_score, _ = _score_lines(base)
    best_points = [p[:] for p in base]
    best_score = base_score
    
    # Multiple restarts
    restarts = 50
    for r in range(restarts):
        if time.time() > deadline - 15:
            break
        
        if r == 0:
            current = [p[:] for p in base]
        else:
            # Perturb the base significantly
            current = [p[:] for p in base]
            for _ in range(rng.randint(5, 15)):
                idx = rng.randint(0, 15)
                new_x = current[idx][0] + rng.randint(-10, 10)
                new_y = current[idx][1] + rng.randint(-10, 10)
                new_pt = [new_x, new_y]
                if any(new_pt == p for p in current):
                    continue
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                current[idx] = new_pt
        
        s, h5 = _score_lines(current)
        if h5:
            continue
        
        opt_pts, opt_score = _local_search(current, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final verification
    final_score, final_h5 = _score_lines(best_points)
    if final_h5 or final_score == 0:
        best_points = [p[:] for p in base]
    
    return best_points
# EVOLVE-BLOCK-END
