# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize a line equation Ax + By + C = 0 to canonical form."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_incidence_for_line(A, B, C, points):
    """Count how many points lie on the line Ax + By + C = 0."""
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _get_all_lines_and_counts(points):
    """
    Generate all lines through pairs of points and count incidences.
    Returns:
        lines_dict: { (A,B,C): count }
        point_lines: list of sets of line keys for each point
    """
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    # Generate lines from all pairs
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            key = _normalize_line(A, B, C)
            if key is not None:
                if key not in lines_dict:
                    lines_dict[key] = 0
                point_lines[i].add(key)
                point_lines[j].add(key)
                
    # Count incidences accurately for each line using vectorized ops
    if lines_dict:
        keys = list(lines_dict.keys())
        A_arr = np.array([k[0] for k in keys], dtype=np.int64)
        B_arr = np.array([k[1] for k in keys], dtype=np.int64)
        C_arr = np.array([k[2] for k in keys], dtype=np.int64)
        xs = points[:, 0].astype(np.int64)
        ys = points[:, 1].astype(np.int64)
        vals = A_arr[:, None] * xs[None, :] + B_arr[:, None] * ys[None, :] + C_arr[:, None]
        counts = np.sum(vals == 0, axis=1)
        for idx, key in enumerate(keys):
            lines_dict[key] = int(counts[idx])
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    """Return sum of top k values in a dict."""
    if len(counts) < k:
        return sum(counts.values())
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _top_k_sum_from_array(arr, k=27):
    """Return sum of top k values in a numpy array."""
    if len(arr) < k:
        return int(np.sum(arr))
    top_k = np.partition(arr, -k)[-k:]
    return int(np.sum(top_k))

def _select_top_lines(lines_dict, k=27):
    """Select top k lines by incidence count."""
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    if len(selected) < k:
        # Pad with dummy lines that have 0 incidence
        sel_keys = set(selected)
        for key in lines_dict:
            if len(selected) >= k:
                break
            if key not in sel_keys:
                selected.append(key)
        while len(selected) < k:
            selected.append((1, 0, 1000000 + len(selected)))
    return np.array(selected, dtype=np.int64)

def _evaluate_removal(base_points, num_remove, deadline):
    """
    Evaluate removing num_remove points from base_points to maximize top-27 line incidences.
    Uses incremental updates for efficiency.
    """
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_mask = None
    
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        if score > best_score:
            best_score = score
            best_mask = np.ones(n, dtype=bool)
        return best_score, base_points[best_mask] if best_mask is not None else None
    
    if num_remove == 1:
        # Vectorized: for each point i, compute the effect of removing it
        keys = list(lines_dict.keys())
        counts_arr = np.array([lines_dict[k] for k in keys], dtype=np.int64)
        
        for i in range(n):
            if time.time() > deadline:
                break
            temp_counts = counts_arr.copy()
            for key in point_lines[i]:
                idx = keys.index(key)
                temp_counts[idx] -= 1
            score = _top_k_sum_from_array(temp_counts)
            if score > best_score:
                best_score = score
                best_mask = np.ones(n, dtype=bool)
                best_mask[i] = False
                
    elif num_remove == 2:
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                temp_counts = dict(lines_dict)
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = _top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    best_mask = np.ones(n, dtype=bool)
                    best_mask[i] = False
                    best_mask[j] = False
    
    elif num_remove == 3:
        # Try removing triples - only feasible for small n
        if n <= 60:
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            break
                        temp_counts = dict(lines_dict)
                        for key in point_lines[i]:
                            temp_counts[key] -= 1
                        for key in point_lines[j]:
                            temp_counts[key] -= 1
                        for key in point_lines[k]:
                            temp_counts[key] -= 1
                        score = _top_k_sum(temp_counts)
                        if score > best_score:
                            best_score = score
                            best_mask = np.ones(n, dtype=bool)
                            best_mask[i] = False
                            best_mask[j] = False
                            best_mask[k] = False
    
    if best_mask is None:
        return -1, None
    return best_score, base_points[best_mask]

def _generate_grid(a, b):
    """Generate an a x b grid of points."""
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _try_grid_with_removal(a, b, num_remove, deadline):
    """Try a grid of size a*b, removing num_remove points to get 54."""
    if a * b != 54 + num_remove:
        return None, None
    pts_base = _generate_grid(a, b)
    score, pts_best = _evaluate_removal(pts_base, num_remove, deadline)
    if pts_best is not None and score > 0:
        lines_dict, _ = _get_all_lines_and_counts(pts_best)
        lines_arr = _select_top_lines(lines_dict)
        return pts_best, lines_arr
    return None, None

def _try_remove_multiple(base_points, num_remove, deadline):
    """
    Try removing num_remove points using a greedy approach:
    iteratively remove the point whose removal causes the least damage to top-27 sum.
    """
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    pts = base_points.copy()
    mask = np.ones(n, dtype=bool)
    
    for _ in range(num_remove):
        if time.time() > deadline:
            break
        current_lines, current_point_lines = _get_all_lines_and_counts(pts[mask])
        current_score = _top_k_sum(current_lines)
        
        best_removal = -1
        best_new_score = -1
        
        for i in range(len(pts[mask])):
            if time.time() > deadline:
                break
            temp_counts = dict(current_lines)
            for key in current_point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_new_score:
                best_new_score = score
                best_removal = i
        
        if best_removal >= 0:
            mask_indices = np.where(mask)[0]
            actual_idx = mask_indices[best_removal]
            mask[actual_idx] = False
        else:
            break
    
    final_pts = pts[mask]
    lines_dict, _ = _get_all_lines_and_counts(final_pts)
    score = _top_k_sum(lines_dict)
    return score, final_pts

def _refine_lines(points, lines, deadline):
    """
    Try to improve the set of lines by swapping individual lines.
    """
    n = points.shape[0]
    lines_dict, _ = _get_all_lines_and_counts(points)
    
    def _total_incidence(pts, lns):
        total = 0
        for i in range(len(pts)):
            for j in range(len(lns)):
                if lns[j, 0] * pts[i, 0] + lns[j, 1] * pts[i, 1] + lns[j, 2] == 0:
                    total += 1
        return total
    
    current_score = _total_incidence(points, lines)
    improved = True
    
    while improved and time.time() < deadline - 30:
        improved = False
        for idx in range(27):
            if time.time() > deadline - 30:
                break
            current_keys = set()
            for j in range(27):
                A, B, C = lines[j]
                key = _normalize_line(int(A), int(B), int(C))
                if key:
                    current_keys.add(key)
            
            for key, count in lines_dict.items():
                if key in current_keys:
                    continue
                new_lines = lines.copy()
                new_lines[idx] = key
                new_score = _total_incidence(points, new_lines)
                if new_score > current_score:
                    lines = new_lines
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return lines, current_score

def _try_exact_grid(a, b, deadline):
    """Try an exact a*b = 54 grid."""
    if a * b != 54:
        return None, None
    pts = _generate_grid(a, b)
    lines_dict, _ = _get_all_lines_and_counts(pts)
    lines_arr = _select_top_lines(lines_dict)
    return pts, lines_arr

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses systematic search over grid configurations with point removals.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, lines, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = lines
    
    # Phase 1: Try exact grids (a*b = 54)
    exact_grids = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    for a, b in exact_grids:
        if time.time() > deadline - 30:
            break
        pts, lines = _try_exact_grid(a, b, deadline)
        if pts is not None:
            # Compute actual score
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # Phase 2: Try near-rectangular grids with removals
    # 7*8 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(7, 8, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 8*7 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(8, 7, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 5*11 = 55, remove 1
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(5, 11, 1, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 11*5 = 55, remove 1
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(11, 5, 1, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 4*14 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(4, 14, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 14*4 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(14, 4, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 3*19 = 57, remove 3
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(3, 19, 3, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 19*3 = 57, remove 3
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(19, 3, 3, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 2*28 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(2, 28, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # 28*2 = 56, remove 2
    if time.time() < deadline - 60:
        pts, lines = _try_grid_with_removal(28, 2, 2, deadline)
        if pts is not None:
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines[j, 0] * pts[i, 0] + lines[j, 1] * pts[i, 1] + lines[j, 2] == 0:
                        I += 1
            update_best(pts, lines, I)
    
    # New: 6*10 = 60, remove 6 (greedy removal)
    if time.time() < deadline - 120:
        pts_base = _generate_grid(6, 10)
        score, pts_best = _try_remove_multiple(pts_base, 6, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # New: 10*6 = 60, remove 6
    if time.time() < deadline - 120:
        pts_base = _generate_grid(10, 6)
        score, pts_best = _try_remove_multiple(pts_base, 6, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # New: 5*12 = 60, remove 6
    if time.time() < deadline - 120:
        pts_base = _generate_grid(5, 12)
        score, pts_best = _try_remove_multiple(pts_base, 6, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # New: 12*5 = 60, remove 6
    if time.time() < deadline - 120:
        pts_base = _generate_grid(12, 5)
        score, pts_best = _try_remove_multiple(pts_base, 6, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # New: 9*7 = 63, remove 9
    if time.time() < deadline - 180:
        pts_base = _generate_grid(9, 7)
        score, pts_best = _try_remove_multiple(pts_base, 9, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # New: 7*9 = 63, remove 9
    if time.time() < deadline - 180:
        pts_base = _generate_grid(7, 9)
        score, pts_best = _try_remove_multiple(pts_base, 9, deadline)
        if pts_best is not None and len(pts_best) == 54:
            lines_dict, _ = _get_all_lines_and_counts(pts_best)
            lines_arr = _select_top_lines(lines_dict)
            I = 0
            for i in range(54):
                for j in range(27):
                    if lines_arr[j, 0] * pts_best[i, 0] + lines_arr[j, 1] * pts_best[i, 1] + lines_arr[j, 2] == 0:
                        I += 1
            update_best(pts_best, lines_arr, I)
    
    # Phase 3: Refinement - try swapping points in the best configuration
    if best_points is not None and time.time() < deadline - 120:
        # Try replacing each point with nearby lattice points
        pts = best_points.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            lines_dict, _ = _get_all_lines_and_counts(pts)
            point_lines = [set() for _ in range(54)]
            for i in range(54):
                for j in range(i + 1, 54):
                    p1 = pts[i]
                    p2 = pts[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    A = dy
                    B = -dx
                    C = dx * p1[1] - dy * p1[0]
                    key = _normalize_line(A, B, C)
                    if key is not None:
                        point_lines[i].add(key)
                        point_lines[j].add(key)
            
            for idx in range(54):
                if time.time() > deadline - 30:
                    break
                # Try moving this point to nearby positions
                x0, y0 = pts[idx]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        new_x, new_y = x0 + dx, y0 + dy
                        if new_x < 1 or new_y < 1:
                            continue
                        # Check if new position is already occupied
                        if np.any((pts[:, 0] == new_x) & (pts[:, 1] == new_y)):
                            continue
                        
                        # Compute new score incrementally
                        # Remove old point's contributions
                        temp_counts = dict(lines_dict)
                        for key in point_lines[idx]:
                            if key in temp_counts:
                                temp_counts[key] -= 1
                        
                        # Add new point's contributions
                        new_point = np.array([[new_x, new_y]])
                        for j in range(54):
                            if j == idx:
                                continue
                            p1 = new_point[0]
                            p2 = pts[j]
                            dx2 = p2[0] - p1[0]
                            dy2 = p2[1] - p1[1]
                            A = dy2
                            B = -dx2
                            C = dx2 * p1[1] - dy2 * p1[0]
                            key = _normalize_line(A, B, C)
                            if key is not None:
                                if key not in temp_counts:
                                    temp_counts[key] = 0
                                temp_counts[key] += 1
                        
                        score = _top_k_sum(temp_counts)
                        if score > current_score:
                            # Apply the move
                            pts[idx] = [new_x, new_y]
                            current_score = score
                            improved = True
                            # Recompute lines_dict and point_lines
                            lines_dict, _ = _get_all_lines_and_counts(pts)
                            point_lines = [set() for _ in range(54)]
                            for i in range(54):
                                for j in range(i + 1, 54):
                                    p1 = pts[i]
                                    p2 = pts[j]
                                    dx3 = p2[0] - p1[0]
                                    dy3 = p2[1] - p1[1]
                                    A = dy3
                                    B = -dx3
                                    C = dx3 * p1[1] - dy3 * p1[0]
                                    key = _normalize_line(A, B, C)
                                    if key is not None:
                                        point_lines[i].add(key)
                                        point_lines[j].add(key)
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                update_best(pts, _select_top_lines(lines_dict), current_score)
    
    # Phase 4: Line refinement - try swapping lines for better incidence
    if best_points is not None and best_lines is not None and time.time() < deadline - 60:
        refined_lines, refined_score = _refine_lines(best_points, best_lines, deadline)
        if refined_score > best_score:
            update_best(best_points, refined_lines, refined_score)
    
    # Final validation and return
    if best_points is None:
        # Fallback to 6x9 grid
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_list = []
        for c in range(1, 7):
            lines_list.append((1, 0, -c))
        for c in range(1, 10):
            lines_list.append((0, 1, -c))
        for c in [0, 1, 2, 3]:
            lines_list.append((1, -1, c))
        for c in [7, 8, 9, 10]:
            lines_list.append((1, 1, -c))
        for c in [4, -1]:
            lines_list.append((1, -1, c))
        for c in [6, 11]:
            lines_list.append((1, 1, -c))
        best_lines = np.array(lines_list, dtype=np.int64)
    else:
        best_lines = _select_top_lines(_get_all_lines_and_counts(best_points)[0])
    
    return best_points, best_lines

# EVOLVE-BLOCK-END
