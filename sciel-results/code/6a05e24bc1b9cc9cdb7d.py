import time
import random
import numpy as np
from collections import Counter

def _make_grid(cols, rows, x0=0, y0=0):
    """Generate a cols x rows grid of points starting at (x0, y0)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((x0 + c, y0 + r))
    return pts

def _make_sheared_grid(cols, rows, s, x0=0, y0=0):
    """
    Generate a sheared grid.
    Basis vectors: (1, 0) and (s, 1).
    Points: (x0 + c + s*r, y0 + r) for c in [0, cols), r in [0, rows).
    """
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = x0 + c + s * r
            y = y0 + r
            pts.append((x, y))
    return pts

def _full_score_and_target(points):
    """Compute full histogram, return (max_count, target_area)."""
    n = len(points)
    pts_arr = np.array(points, dtype=np.int64)
    counts = Counter()
    for i in range(n):
        xi, yi = int(pts_arr[i][0]), int(pts_arr[i][1])
        for j in range(i+1, n):
            xj, yj = int(pts_arr[j][0]), int(pts_arr[j][1])
            for k in range(j+1, n):
                xk, yk = int(pts_arr[k][0]), int(pts_arr[k][1])
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
    """
    n = len(points)
    old_pt = points[idx]
    delta = 0
    
    for j in range(n):
        if j == idx: continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx: continue
            xk, yk = points[k]
            
            d_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (yj - old_pt[1]) * (xk - old_pt[0]))
            d_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (yj - new_pt[1]) * (xk - new_pt[0]))
            
            if d_old == target_area and d_old > 0:
                delta -= 1
            if d_new == target_area and d_new > 0:
                delta += 1
                
    return delta

def _delta_count_for_swap(points, idx1, idx2, target_area):
    """
    Calculate the change in the count of triangles with 'target_area'
    when points[idx1] and points[idx2] are swapped.
    """
    n = len(points)
    p1 = points[idx1]
    p2 = points[idx2]
    delta = 0
    
    for j in range(n):
        if j == idx1 or j == idx2: continue
        xj, yj = points[j]
        for k in range(j+1, n):
            if k == idx1 or k == idx2: continue
            xk, yk = points[k]
            
            d_old1 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            d_new1 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            if d_old1 == target_area and d_old1 > 0:
                delta -= 1
            if d_new1 == target_area and d_new1 > 0:
                delta += 1
            
            d_old2 = abs((xj - p2[0]) * (yk - p2[1]) - (yj - p2[1]) * (xk - p2[0]))
            d_new2 = abs((xj - p1[0]) * (yk - p1[1]) - (yj - p1[1]) * (xk - p1[0]))
            if d_old2 == target_area and d_old2 > 0:
                delta -= 1
            if d_new2 == target_area and d_new2 > 0:
                delta += 1
    
    return delta

def construct():
    start_time = time.time()
    deadline = start_time + 55

    # 1. Generate Candidates
    candidates = []
    
    # Rectangular Grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10), (15, 2), (2, 15)]:
        # Center the grid
        x0 = -cols // 2
        y0 = -rows // 2
        pts = _make_grid(cols, rows, x0, y0)
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            candidates.append(pts)
            
    # Sheared Grids
    for cols, rows in [(6, 5), (5, 6), (10, 3), (3, 10)]:
        for s in [-2, -1, 1, 2]:
            x0 = -cols // 2 - s * (rows // 2)
            y0 = -rows // 2
            pts = _make_sheared_grid(cols, rows, s, x0, y0)
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
                candidates.append(pts)

    # 2. Evaluate Candidates
    best_pts = None
    best_count = -1
    target_area = 1
    
    for pts in candidates:
        if len(set(pts)) != 30: continue
        c, t = _full_score_and_target(pts)
        if c > best_count:
            best_count = c
            target_area = t
            best_pts = [p for p in pts]

    if best_pts is None:
        # Fallback
        best_pts = _make_grid(6, 5, -2, -2)
        best_count, target_area = _full_score_and_target(best_pts)

    random.seed(42)
    no_improve_counter = 0
    
    while time.time() < deadline:
        if no_improve_counter > 100:
            # Re-evaluate target area
            c, t = _full_score_and_target(best_pts)
            if c > best_count:
                best_count = c
                target_area = t
            no_improve_counter = 0
            continue

        improved = False
        use_swap = random.random() < 0.3
        
        if use_swap:
            idx1 = random.randint(0, 29)
            idx2 = random.randint(0, 29)
            if idx1 != idx2:
                delta = _delta_count_for_swap(best_pts, idx1, idx2, target_area)
                if delta > 0:
                    best_pts[idx1], best_pts[idx2] = best_pts[idx2], best_pts[idx1]
                    best_count += delta
                    no_improve_counter = 0
                    improved = True
        else:
            idx = random.randint(0, 29)
            orig = best_pts[idx]
            # Try a few random moves and local neighborhood
            # Local neighborhood
            radius = 3
            best_delta = 0
            best_new_pt = None
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0: continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30): continue
                    if (nx, ny) in best_pts: continue
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = (nx, ny)
            
            if best_new_pt and best_delta > 0:
                best_pts[idx] = best_new_pt
                best_count += best_delta
                no_improve_counter = 0
                improved = True
            else:
                # Try random jump
                for _ in range(5):
                    nx = random.randint(-30, 30)
                    ny = random.randint(-30, 30)
                    if (nx, ny) in best_pts: continue
                    delta = _delta_count_for_target(best_pts, idx, (nx, ny), target_area)
                    if delta > 0:
                        best_pts[idx] = (nx, ny)
                        best_count += delta
                        no_improve_counter = 0
                        improved = True
                        break

        if not improved:
            no_improve_counter += 1
            
        # Occasional restart
        if no_improve_counter > 200 and random.random() < 0.1:
             idx = random.randint(0, 29)
             nx = random.randint(-30, 30)
             ny = random.randint(-30, 30)
             if (nx, ny) not in best_pts:
                 best_pts[idx] = (nx, ny)
                 best_count, target_area = _full_score_and_target(best_pts)
                 no_improve_counter = 0

    return [list(p) for p in best_pts]
