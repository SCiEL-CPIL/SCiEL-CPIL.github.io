import numpy as np
import time


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _max_grow_room(centers, sides, i, tol=1e-7):
    """Compute maximum additional side length square i can grow by."""
    n = len(sides)
    s = sides[i]
    cx, cy = centers[i]
    
    room_left = cx - s/2
    room_right = 1 - (cx + s/2)
    room_bottom = cy - s/2
    room_top = 1 - (cy + s/2)
    
    max_boundary = min(room_left, room_right, room_bottom, room_top)
    if max_boundary < 0:
        return 0.0
    
    max_room = max_boundary
    for j in range(n):
        if j == i:
            continue
        s_j = sides[j]
        cx_j, cy_j = centers[j]
        
        dx = abs(cx - cx_j)
        dy = abs(cy - cy_j)
        current_t = (s + s_j)/2 - tol
        
        if dx >= current_t or dy >= current_t:
            min_sep = min(dx, dy)
            ds_limit = 2 * (min_sep + tol - (s + s_j)/2)
            if ds_limit < max_room:
                max_room = max(0.0, ds_limit)
        else:
            max_room = 0.0
            break
    
    return max(0.0, max_room)


def _local_refine(centers, sides, tol=1e-7, max_iter=5000, deadline=None):
    """
    Local refinement: grow squares and adjust positions to maximize total side length.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if deadline is None:
        deadline = time.time() + 100
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        improved = False
        
        for i in range(n):
            room = _max_grow_room(best_centers, best_sides, i, tol)
            if room > 1e-10:
                for frac in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
                    new_sides = best_sides.copy()
                    new_sides[i] = best_sides[i] + room * frac
                    new_centers = best_centers.copy()
                    
                    s = new_sides[i]
                    cx = np.clip(best_centers[i, 0], s/2, 1 - s/2)
                    cy = np.clip(best_centers[i, 1], s/2, 1 - s/2)
                    new_centers[i] = [cx, cy]
                    
                    if _is_feasible(new_centers, new_sides, tol):
                        best_centers = new_centers
                        best_sides = new_sides
                        best_val = np.sum(best_sides)
                        improved = True
                        break
        
        for i in range(n):
            s = best_sides[i]
            cx, cy = best_centers[i]
            
            for dx in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                for dy in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                    new_centers = best_centers.copy()
                    new_cx = np.clip(cx + dx, s/2, 1 - s/2)
                    new_cy = np.clip(cy + dy, s/2, 1 - s/2)
                    new_centers[i] = [new_cx, new_cy]
                    
                    if _is_feasible(new_centers, best_sides, tol):
                        old_room = _max_grow_room(best_centers, best_sides, i, tol)
                        new_room = _max_grow_room(new_centers, best_sides, i, tol)
                        if new_room > old_room + 1e-8:
                            best_centers = new_centers
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_centers, best_sides


def _make_alternative_tiling():
    """Alternative: 2 squares of 0.5 on left, 8 squares of 0.25 on right."""
    centers = np.array([
        [0.25, 0.25],
        [0.25, 0.75],
        [0.625, 0.125], [0.875, 0.125],
        [0.625, 0.375], [0.875, 0.375],
        [0.625, 0.625], [0.875, 0.625],
        [0.625, 0.875], [0.875, 0.875],
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 575
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    initial_configs = [
        _make_2big_8small(),
        _make_alternative_tiling(),
    ]
    
    for c, s in initial_configs:
        if time.time() > deadline - 20:
            break
        if _is_feasible(c, s):
            val = np.sum(s)
            if val > best_val:
                best_val = val
                best_centers = c.copy()
                best_sides = s.copy()
    
    if best_centers is None:
        best_centers, best_sides = _make_2big_8small()
        best_val = np.sum(best_sides)
    
    if time.time() < deadline - 10:
        try:
            refine_deadline = min(deadline - 5, time.time() + 200)
            refined_centers, refined_sides = _local_refine(
                best_centers, best_sides, max_iter=5000, deadline=refine_deadline
            )
            if _is_feasible(refined_centers, refined_sides):
                refined_val = np.sum(refined_sides)
                if refined_val > best_val:
                    best_val = refined_val
                    best_centers = refined_centers
                    best_sides = refined_sides
        except Exception:
            pass
    
    if time.time() < deadline - 15:
        try:
            refine_deadline2 = min(deadline - 5, time.time() + 100)
            refined_centers2, refined_sides2 = _local_refine(
                best_centers, best_sides, max_iter=3000, deadline=refine_deadline2
            )
            if _is_feasible(refined_centers2, refined_sides2):
                refined_val2 = np.sum(refined_sides2)
                if refined_val2 > best_val:
                    best_val = refined_val2
                    best_centers = refined_centers2
                    best_sides = refined_sides2
        except Exception:
            pass
    
    return best_centers, best_sides
