import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    counts = Counter()
    # Use numpy for faster determinant calculation if possible, but N=30 is small
    # Pure python with optimized loops is fine
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                d = abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0, 0
    target_area, max_count = counts.most_common(1)[0]
    return max_count, target_area

def _delta_count_for_target(points, idx, new_pt, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx] is replaced by new_pt.
    Optimized by pre-filtering 'other' points.
    """
    old_pt = points[idx]
    others = [points[j] for j in range(len(points)) if j != idx]
    
    delta = 0
    n_others = len(others)
    
    # Iterate over all pairs in 'others'
    for j in range(n_others):
        xj, yj = others[j]
        for k in range(j+1, n_others):
            xk, yk = others[k]
            
            # Old area
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            # New area
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area:
                delta -= 1
            if d_new == target_area:
                delta += 1
                
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Try several initial configurations
    best_pts = None
    best_count = 0
    target_area = 0
    
    configs = [
        (6, 5, -2, -2),
        (5, 6, -2, -2),
        (10, 3, -4, -1),
        (3, 10, -1, -4),
        (15, 2, -7, 0),
        (2, 15, 0, -7),
        (1, 30, -14, 0), # Degenerate, will be 0 area, but valid input
        (30, 1, 0, -14), # Degenerate
    ]
    
    for cols, rows, x0, y0 in configs:
        if cols * rows != 30:
            continue
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            c, t = _full_score_and_target(pts)
            if c > best_count:
                best_count = c
                target_area = t
                best_pts = pts[:]
                
    if best_pts is None:
        # Fallback random
        best_pts = []
        while len(best_pts) < 30:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in best_pts:
                best_pts.append(p)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    
    current_pts = best_pts[:]
    current_count = best_count
    current_target = target_area
    
    # We track the "score" as the count of the current target area.
    # We must be careful: if the global max area changes, our 'current_count' might not reflect the true score.
    # However, if we only accept moves that improve 'current_count' (or accept worse with prob),
    # we are effectively optimizing for that specific area.
    # To ensure we don't miss a better area, we re-evaluate periodically.
    
    last_reval_time = start_time
    steps = 0
    
    while time.time() < deadline:
        # Periodic re-evaluation to update target and check global best
        if time.time() - last_reval_time > 2.0:
            c, t = _full_score_and_target(current_pts)
            if c > best_count:
                best_count = c
                best_pts = current_pts[:]
            # Update target if the current target is no longer the max, or if a new max is found
            # Actually, if c > current_count, it's possible the max area changed.
            # Let's just update best_pts if we found a better global score.
            # For the SA loop, we stick to the current_target to keep the delta calculation simple and fast.
            # If the current_target is no longer the best, the SA might be optimizing the wrong thing.
            # So, if the max area in the histogram is different from current_target, switch target.
            # We can get the top 2 from _full_score? No, let's just re-eval and take the max.
            # If the max area changed, we should update current_target to the new max.
            # But we don't know the count of the *new* target for the *current* points easily without re-eval.
            # The re-eval gives us the max count c and max area t.
            # So we can set current_target = t and current_count = c.
            current_target = t
            current_count = c
            last_reval_time = time.time()
            T = max(T, 5.0) # Reheat slightly on re-eval to allow exploration
            
        # Propose a move
        # 50% chance small move, 50% chance large move
        if random.random() < 0.5:
            # Small move
            idx = random.randint(0, 29)
            orig = current_pts[idx]
            radius = 3
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in current_pts:
                            candidates.append((nx, ny))
            if not candidates:
                continue
            new_pt = random.choice(candidates)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        else:
            # Large move (random jump)
            idx = random.randint(0, 29)
            nx = random.randint(-30, 30)
            ny = random.randint(-30, 30)
            if (nx, ny) in current_pts:
                continue
            new_pt = (nx, ny)
            delta = _delta_count_for_target(current_pts, idx, new_pt, current_target)
            
            # Acceptance
            if delta > 0 or random.random() < np.exp(delta / T):
                current_pts[idx] = new_pt
                current_count += delta
                if current_count > best_count:
                    best_count = current_count
                    best_pts = current_pts[:]
        
        T *= cooling
        if T < min_T:
            T = min_T
            # If T is too low, we are stuck in local optimum.
            # Try a random restart from best_pts with some perturbation?
            # Or just increase T slightly.
            T = 5.0
            
        steps += 1

    # Final check
    c, t = _full_score_and_target(best_pts)
    if c > best_count:
        best_count = c
        best_pts = [list(p) for p in best_pts] # Ensure list of lists

    return [list(p) for p in best_pts]
