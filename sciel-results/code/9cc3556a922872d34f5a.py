import numpy as np
import time
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _get_all_lines_and_counts(points):
    n = points.shape[0]
    lines_dict = {}
    
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
                lines_dict[key] += 1
                
    # Convert pair counts to point counts
    # If a line has k points, it appears C(k,2) times.
    # We need to find k such that C(k,2) = count? No, that's not right.
    # The count in lines_dict is the number of pairs.
    # We need the number of points.
    # It is faster to just re-count points for each unique line.
    for key in lines_dict:
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        lines_dict[key] = int(np.sum(vals == 0))
        
    return lines_dict

def _evaluate_configuration(points):
    lines_dict = _get_all_lines_and_counts(points)
    if not lines_dict:
        return 0
    counts = list(lines_dict.values())
    counts.sort(reverse=True)
    k = min(27, len(counts))
    return sum(counts[:k])

def _get_top_lines(points, k=27):
    lines_dict = _get_all_lines_and_counts(points)
    if not lines_dict:
        return np.zeros((k, 3), dtype=np.int64)
    
    sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
    selected = [l for l, c in sorted_lines[:k]]
    
    if len(selected) < k:
        # Pad with arbitrary lines if not enough (unlikely)
        all_keys = list(lines_dict.keys())
        sel_keys = set(selected)
        pad_keys = [key for key in all_keys if key not in sel_keys]
        while len(pad_keys) < k - len(selected):
            pad_keys.append((1, 0, 1000000 + len(pad_keys)))
        selected.extend(pad_keys[:k - len(selected)])
        
    return np.array(selected, dtype=np.int64)

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _local_search(points, max_rounds=20, time_budget=300, radius=5):
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    start_time = time.time()
    
    for r in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        n = len(current_points)
        occupied = set(map(tuple, current_points))
        
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            
            orig_pt = current_points[i]
            best_move = None
            best_score = current_score
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (int(orig_pt[0] + dx), int(orig_pt[1] + dy))
                    if cand in occupied:
                        continue
                    
                    trial_points = current_points.copy()
                    trial_points[i] = cand
                    trial_score = _evaluate_configuration(trial_points)
                    
                    if trial_score > best_score:
                        best_score = trial_score
                        best_move = cand
            
            if best_move is not None:
                old_pt = tuple(current_points[i])
                occupied.remove(old_pt)
                current_points[i] = best_move
                occupied.add(best_move)
                current_score = best_score
                improved = True
        
        if not improved:
            break
    
    return current_score, current_points

def _swap_search(points, max_rounds=20, time_budget=300, radius=8):
    current_points = points.copy()
    current_score = _evaluate_configuration(current_points)
    start_time = time.time()
    
    for r in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        n = len(current_points)
        occupied = set(map(tuple, current_points))
        
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            
            orig_pt = current_points[i]
            best_swap = None
            best_score = current_score
            
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (int(orig_pt[0] + dx), int(orig_pt[1] + dy))
                    if cand in occupied:
                        continue
                    
                    trial_points = current_points.copy()
                    trial_points[i] = cand
                    trial_score = _evaluate_configuration(trial_points)
                    
                    if trial_score > best_score:
                        best_score = trial_score
                        best_swap = cand
            
            if best_swap is not None:
                old_pt = tuple(current_points[i])
                occupied.remove(old_pt)
                current_points[i] = best_swap
                occupied.add(best_swap)
                current_score = best_score
                improved = True
        
        if not improved:
            break
    
    return current_score, current_points

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            best_lines = _get_top_lines(pts, 27)
            
    # Grid configurations
    # Focus on compact grids that allow for good diagonal incidences
    # 8x7 = 56 (remove 2)
    # 9x6 = 54 (remove 0)
    # 6x9 = 54 (remove 0)
    # 7x8 = 56 (remove 2)
    # 5x11 = 55 (remove 1)
    # 11x5 = 55 (remove 1)
    
    grid_configs = [
        (9, 6),
        (6, 9),
        (8, 7),
        (7, 8),
        (5, 11),
        (11, 5),
        (4, 14),
        (14, 4),
        (10, 6), # 60, remove 6
        (6, 10), # 60, remove 6
    ]
    
    for a, b in grid_configs:
        if time.time() > deadline - 30:
            break
        
        pts = _generate_grid(a, b)
        
        if len(pts) > 54:
            # Remove points farthest from center
            center_x = (a + 1) / 2.0
            center_y = (b + 1) / 2.0
            distances = np.sqrt((pts[:, 0] - center_x)**2 + (pts[:, 1] - center_y)**2)
            order = np.argsort(distances)[::-1]
            R = len(pts) - 54
            remove_indices = order[:R]
            mask = np.ones(len(pts), dtype=bool)
            mask[remove_indices] = False
            pts = pts[mask]
        elif len(pts) < 54:
            continue
            
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
        # Local search
        if time.time() < deadline - 20:
            remaining = deadline - time.time() - 10
            budget = min(60, remaining)
            if budget > 5:
                new_score, new_pts = _local_search(pts, max_rounds=10, time_budget=budget, radius=4)
                if new_score > best_score:
                    update_best(new_pts, new_score)
                    
    # Second pass with more aggressive search on the best found so far
    if best_points is not None and time.time() < deadline - 30:
        remaining = deadline - time.time() - 10
        budget = min(120, remaining)
        if budget > 10:
            new_score, new_pts = _swap_search(best_points, max_rounds=15, time_budget=budget, radius=6)
            if new_score > best_score:
                update_best(new_pts, new_score)
                
    if best_points is None:
        pts = _generate_grid(9, 6)
        score = _evaluate_configuration(pts)
        update_best(pts, score)
        
    return best_points, best_lines
