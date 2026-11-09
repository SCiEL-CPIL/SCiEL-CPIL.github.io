import numpy as np
import time
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_key(A, B, C):
    return _normalize_line(A, B, C)

def _count_incidences(points, lines):
    if len(points) == 0 or len(lines) == 0:
        return 0
    # points: (N, 2), lines: (M, 3)
    # val[i, j] = A_i * x_j + B_i * y_j + C_i
    val = (lines[:, 0:1] * points[:, 0:1].T + 
           lines[:, 1:2] * points[:, 1:2].T + 
           lines[:, 2:3])
    return int(np.sum(val == 0))

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _line_key(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _generate_initial_grid(rows, cols, total_points=54):
    """Generate a grid of points. If rows*cols > total_points, take a subset? 
    No, we need exactly 54. 
    If rows*cols == 54, take all.
    If rows*cols < 54, we can't. So we must pick rows, cols such that rows*cols >= 54?
    Actually, the task says 54 points. A grid of R x C has R*C points.
    If R*C != 54, we have to either add/remove points or use a partial grid.
    The current program uses _generate_grid_points which takes the first 54 in row-major order.
    This creates a "rectangular" block of points.
    Let's stick to that for initialization, but allow local search to move them.
    """
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append((c, r)) # (x, y)
            if len(pts) == total_points:
                break
        if len(pts) == total_points:
            break
    if len(pts) < total_points:
        # Pad with nearby points? Or just return what we have? 
        # The shape must be (54, 2). 
        # If we can't fill it, we should pick a grid size that fills it or pads it.
        # For now, assume we pick grid sizes where we can fill 54.
        # E.g., 9x6=54, 6x9=54, 18x3=54, 3x18=54, 27x2=54, 2x27=54.
        # Or 10x6=60, take first 54.
        pass
    return np.array(pts, dtype=np.int64)

def _get_candidate_lines(points, max_candidates=200):
    """Generate a pool of candidate lines based on the current points.
    We prioritize lines that pass through many points.
    """
    n = len(points)
    candidates = {}
    
    # 1. Lines through pairs of points
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _line_key(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            if key not in candidates:
                candidates[key] = 0
                
    # 2. Also consider some "structural" lines near the grid
    # The points are roughly in a grid. Let's add lines that are close to grid lines.
    # Actually, the pair-generation should cover most useful lines.
    # Let's sort by incidence count and keep the top ones.
    
    line_list = list(candidates.keys())
    if not line_list:
        return np.zeros((max_candidates, 3), dtype=np.int64)
    
    lines_arr = np.array(line_list, dtype=np.int64)
    # Count incidences
    val = (lines_arr[:, 0:1] * points[:, 0:1].T + 
           lines_arr[:, 1:2] * points[:, 1:2].T + 
           lines_arr[:, 2:3])
    counts = np.sum(val == 0, axis=1)
    
    order = np.argsort(-counts)
    top = order[:max_candidates]
    return lines_arr[top]

def _local_search(points, lines, time_limit, max_iterations=10000):
    """
    Perform local search to improve the incidence count.
    Moves:
    1. Move a point to a nearby integer coordinate.
    2. Swap a line with a candidate line from the pool.
    """
    start_time = time.time()
    current_count = _count_incidences(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    best_count = current_count
    
    N = 54
    M = 27
    
    # Precompute incidence matrix for current state
    # Inc[i, j] is 1 if point i is on line j
    def compute_incidence_matrix(pts, lns):
        val = (lns[:, 0:1] * pts[:, 0:1].T + 
               lns[:, 1:2] * pts[:, 1:2].T + 
               lns[:, 2:3])
        return (val == 0).astype(np.int8)
    
    inc_matrix = compute_incidence_matrix(points, lines)
    
    # Candidate lines pool. We will update this occasionally or keep it static for a while.
    # To save time, let's generate a static pool of top 100 lines from the initial points.
    # Then, if a line is not in the pool, we might miss it. 
    # But moving points changes which lines are good.
    # Strategy: Every 100 iterations, refresh the candidate pool.
    
    candidate_pool = _get_candidate_lines(points, 100)
    candidate_set = {tuple(l) for l in candidate_pool}
    
    for it in range(max_iterations):
        if time.time() > time_limit:
            break
            
        # 1. Try moving a point
        # Pick a random point
        i = np.random.randint(N)
        old_pt = points[i].copy()
        
        # Try moving to a nearby point
        # Neighbors: (x+dx, y+dy) for dx, dy in [-2, 2]
        best_delta = 0
        best_new_pt = None
        
        # Calculate current contribution of point i
        # If we move point i, the incidence count changes by:
        # - (number of lines currently incident to i)
        # + (number of lines that would be incident to new_pt)
        # But lines are fixed for this move.
        
        current_inc_count = np.sum(inc_matrix[i, :])
        
        # We need to check which lines the new point would be on.
        # To be fast, we can precompute the "distance" or just check the current lines.
        # Actually, we only care about the current set of lines for the immediate gain.
        
        # Let's check neighbors
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = old_pt + np.array([dx, dy])
                new_x, new_y = int(new_pt[0]), int(new_pt[1])
                
                # Check if new_pt is already occupied
                # This is O(N) check. We can maintain a set of points.
                # For now, let's just check.
                # Optimization: maintain a point_set
                
                # Check incidence with current lines
                # val_j = A_j * new_x + B_j * new_y + C_j
                # We want to count how many are 0.
                
                # Vectorized check
                vals = (lines[:, 0] * new_x + lines[:, 1] * new_y + lines[:, 2])
                new_inc_count = np.sum(vals == 0)
                
                delta = new_inc_count - current_inc_count
                if delta > best_delta:
                    # Check if point is unique
                    # We need to ensure (new_x, new_y) is not in points (excluding i)
                    # This check is expensive if done naively.
                    # Let's assume for a moment we just take the best delta and check validity later.
                    # Or better: maintain a set of points.
                    best_delta = delta
                    best_new_pt = (new_x, new_y)
        
        if best_new_pt is not None and best_delta > 0:
            # Check if best_new_pt is already occupied by another point
            # Convert points to set for O(1) lookup? 
            # Let's do a quick check.
            pt_tuples = [tuple(p) for p in points]
            if best_new_pt in pt_tuples:
                # It's occupied. We would need to swap or ignore.
                # For simplicity, ignore if occupied.
                pass
            else:
                # Apply move
                points[i] = best_new_pt
                # Update incidence matrix for point i
                vals = (lines[:, 0] * best_new_pt[0] + lines[:, 1] * best_new_pt[1] + lines[:, 2])
                inc_matrix[i, :] = (vals == 0).astype(np.int8)
                
                current_count += best_delta
                if current_count > best_count:
                    best_count = current_count
                    best_points = points.copy()
                    best_lines = lines.copy()
        
        # 2. Try swapping a line
        # Pick a random line
        j = np.random.randint(M)
        old_line = lines[j].copy()
        
        # Current incidence count for line j
        current_line_count = np.sum(inc_matrix[:, j])
        
        # Find the best candidate line to replace it
        # We have a candidate_pool.
        # We need to find a line in the pool that is NOT currently in lines,
        # and has a high incidence count with the current points.
        
        # Precompute incidence counts for all candidate pool lines
        # This is expensive if done every iteration.
        # Let's do it every 10 iterations.
        if it % 10 == 0:
            # Compute counts for candidate pool
            # candidate_pool is (P, 3)
            # points is (N, 2)
            if len(candidate_pool) > 0:
                val_pool = (candidate_pool[:, 0:1] * points[:, 0:1].T + 
                            candidate_pool[:, 1:2] * points[:, 1:2].T + 
                            candidate_pool[:, 2:3])
                pool_counts = np.sum(val_pool == 0, axis=1)
            else:
                pool_counts = np.array([])
        else:
            # Reuse last computed counts (approximate)
            pass
        
        if len(pool_counts) > 0:
            # Find the best candidate not in current lines
            current_line_keys = {tuple(l) for l in lines}
            
            best_candidate_idx = -1
            best_candidate_count = 0
            best_candidate_line = None
            
            for k in range(len(candidate_pool)):
                cand_key = tuple(candidate_pool[k])
                if cand_key not in current_line_keys:
                    if pool_counts[k] > best_candidate_count:
                        best_candidate_count = pool_counts[k]
                        best_candidate_idx = k
                        best_candidate_line = candidate_pool[k]
            
            if best_candidate_line is not None:
                delta = best_candidate_count - current_line_count
                if delta > 0:
                    # Apply move
                    lines[j] = best_candidate_line
                    # Update incidence matrix for column j
                    vals = (best_candidate_line[0] * points[:, 0] + 
                            best_candidate_line[1] * points[:, 1] + 
                            best_candidate_line[2])
                    inc_matrix[:, j] = (vals == 0).astype(np.int8)
                    
                    current_count += delta
                    if current_count > best_count:
                        best_count = current_count
                        best_points = points.copy()
                        best_lines = lines.copy()
        
        # Occasionally refresh candidate pool
        if it % 100 == 0:
            candidate_pool = _get_candidate_lines(points, 100)

    return best_points, best_lines, best_count

def szemeredi_trotter_incidences():
    deadline = time.time() + 600
    safety_margin = 30
    
    best_points = None
    best_lines = None
    best_count = 0
    
    # Try different grid shapes
    # 9x6, 6x9, 18x3, 3x18, 10x6 (take 54), etc.
    grid_shapes = [
        (9, 6),
        (6, 9),
        (18, 3),
        (3, 18),
        (27, 2),
        (2, 27),
        (10, 6), # 60 points, take 54
        (6, 10),
    ]
    
    for rows, cols in grid_shapes:
        if time.time() > deadline - safety_margin:
            break
            
        pts = _generate_initial_grid(rows, cols)
        if pts.shape != (54, 2):
            # If shape is wrong, we need to adjust.
            # _generate_initial_grid takes first 54.
            # If rows*cols < 54, it will be shorter.
            # Let's ensure we only use shapes where rows*cols >= 54.
            if rows * cols < 54:
                continue
            # If it's longer, it's truncated to 54.
            # But _generate_initial_grid breaks when len(pts) == 54.
            # So it should be (54, 2).
            pass
        
        # Check uniqueness
        pt_tuples = set(map(tuple, pts))
        if len(pt_tuples) != 54:
            continue
            
        # Get initial lines
        # Use top 27 lines from candidate pool
        cand_lines = _get_candidate_lines(pts, 100)
        lines = cand_lines[:27]
        
        # Ensure lines are valid and unique
        line_set = set()
        valid_lines = []
        for l in lines:
            key = _line_key(l[0], l[1], l[2])
            if key not in line_set and key[0] != 0 and key[1] != 0:
                line_set.add(key)
                valid_lines.append(l)
        # If we don't have 27, pad with arbitrary lines? 
        # Or just use what we have? The shape must be (27, 3).
        # We need exactly 27.
        # Let's generate more candidates if needed.
        if len(valid_lines) < 27:
            # Generate more candidates
            more_cands = _get_candidate_lines(pts, 200)
            for l in more_cands:
                if len(valid_lines) >= 27:
                    break
                key = _line_key(l[0], l[1], l[2])
                if key not in line_set and key[0] != 0 and key[1] != 0:
                    line_set.add(key)
                    valid_lines.append(l)
        lines = np.array(valid_lines[:27], dtype=np.int64)
        
        if lines.shape != (27, 3):
            continue
            
        # Local Search
        t_limit = deadline - safety_margin
        p, l, c = _local_search(pts, lines, t_limit, max_iterations=2000)
        
        if c > best_count:
            best_count = c
            best_points = p.copy()
            best_lines = l.copy()
            
    if best_points is None:
        # Fallback
        pts = _generate_initial_grid(9, 6)
        lines = _get_candidate_lines(pts, 27)[:27]
        best_points = pts
        best_lines = lines
        
    if not _is_valid(best_points, best_lines):
        pts = _generate_initial_grid(9, 6)
        lines = _get_candidate_lines(pts, 27)[:27]
        return pts, lines
        
    return best_points, best_lines
