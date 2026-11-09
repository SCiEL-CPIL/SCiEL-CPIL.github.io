import numpy as np
import time
from math import gcd
from itertools import combinations, product

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

def _count_incidence_for_line(A, B, C, points):
    vals = A * points[:, 0] + B * points[:, 1] + C
    return int(np.sum(vals == 0))

def _get_all_lines_and_counts(points):
    """
    Returns:
    lines_dict: { (A,B,C): count }
    point_lines: list of sets of line keys for each point
    """
    n = points.shape[0]
    lines_dict = {}
    point_lines = [set() for _ in range(n)]
    
    # Generate lines from pairs
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
                
    # Count incidences accurately
    for key in lines_dict:
        A, B, C = key
        lines_dict[key] = _count_incidence_for_line(A, B, C, points)
        
    return lines_dict, point_lines

def _top_k_sum(counts, k=27):
    """
    counts: dict {key: count}
    Returns sum of top k counts.
    """
    if len(counts) < k:
        return sum(counts.values())
    # Sort values descending
    sorted_vals = sorted(counts.values(), reverse=True)
    return sum(sorted_vals[:k])

def _evaluate_removal(base_points, num_remove):
    """
    Evaluate all ways to remove num_remove points from base_points.
    Returns best score and the points.
    """
    n = base_points.shape[0]
    if n < num_remove:
        return -1, None
    
    lines_dict, point_lines = _get_all_lines_and_counts(base_points)
    
    best_score = -1
    best_points = None
    
    # If num_remove is small, we can iterate
    if num_remove == 0:
        score = _top_k_sum(lines_dict)
        return score, base_points
        
    if num_remove == 1:
        for i in range(n):
            # Create a copy of counts
            # Instead of copying the whole dict, we can adjust a list of counts?
            # But we need to re-sort.
            # Number of lines is ~1000. Copying dict is fast.
            temp_counts = dict(lines_dict)
            for key in point_lines[i]:
                temp_counts[key] -= 1
            score = _top_k_sum(temp_counts)
            if score > best_score:
                best_score = score
                # Construct points
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                best_points = base_points[mask]
                
    elif num_remove == 2:
        for i in range(n):
            for j in range(i + 1, n):
                temp_counts = dict(lines_dict)
                # Subtract for i
                for key in point_lines[i]:
                    temp_counts[key] -= 1
                # Subtract for j
                for key in point_lines[j]:
                    temp_counts[key] -= 1
                score = _top_k_sum(temp_counts)
                if score > best_score:
                    best_score = score
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    best_points = base_points[mask]
                    
    else:
        # For larger removals, we might need a heuristic or smaller search space
        # For now, skip or try corners only?
        # Let's try removing the first num_remove points? No, that's bad.
        # We'll rely on num_remove 1 and 2 for now.
        pass
        
    return best_score, best_points

def _generate_grid(a, b):
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    best_score = -1
    best_points = None
    best_lines = None
    
    def update_best(pts, score):
        nonlocal best_score, best_points, best_lines
        if score > best_score:
            best_score = score
            best_points = pts
            # Find top 27 lines for these points
            lines_dict, _ = _get_all_lines_and_counts(pts)
            sorted_lines = sorted(lines_dict.items(), key=lambda x: (-x[1], x[0]))
            selected = [l for l, c in sorted_lines[:27]]
            if len(selected) < 27:
                # Pad if necessary? Task says 27 lines.
                # If we have fewer, we must add arbitrary lines?
                # "lines a (27, 3) array"
                # If we have < 27 distinct lines with incidence > 0, we can add lines with 0 incidence.
                # But we want to maximize I.
                # If we have < 27 lines, we just add dummy lines (e.g., x=0, y=0, etc.)
                # But usually we have plenty.
                pass
            best_lines = np.array(selected, dtype=np.int64)
            if len(best_lines) < 27:
                # Pad with zeros (invalid line A=B=0? No, A=1, B=0, C=1000000)
                # Actually, just use lines with 0 incidence.
                # Find some lines not in selected.
                all_keys = list(lines_dict.keys())
                sel_keys = set(l for l, _ in sorted_lines[:27])
                pad_keys = [k for k in all_keys if k not in sel_keys]
                while len(pad_keys) < 27 - len(best_lines):
                    # Generate a new line
                    pad_keys.append((1, 0, 1000000 + len(pad_keys)))
                best_lines = np.vstack([best_lines, np.array(pad_keys, dtype=np.int64)])
                
    # 1. Try exact grids
    grid_candidates = [
        (6, 9), (9, 6), (3, 18), (18, 3), (2, 27), (27, 2), (1, 54), (54, 1)
    ]
    for a, b in grid_candidates:
        if time.time() > deadline - 60:
            break
        pts = _generate_grid(a, b)
        score, _ = _evaluate_removal(pts, 0)
        update_best(pts, score)
        
    # 2. Try grids with removals
    # 7x8 = 56, remove 2
    if time.time() < deadline - 60:
        pts_base = _generate_grid(7, 8)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            update_best(pts_best, score)
            
    # 8x7 = 56, remove 2 (Same as above, but orientation might matter for line selection? No, symmetric)
    # 5x11 = 55, remove 1
    if time.time() < deadline - 60:
        pts_base = _generate_grid(5, 11)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            update_best(pts_best, score)
            
    # 11x5 = 55, remove 1
    if time.time() < deadline - 60:
        pts_base = _generate_grid(11, 5)
        score, pts_best = _evaluate_removal(pts_base, 1)
        if pts_best is not None:
            update_best(pts_best, score)
            
    # 4x14 = 56, remove 2
    if time.time() < deadline - 60:
        pts_base = _generate_grid(4, 14)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            update_best(pts_best, score)
            
    # 14x4 = 56, remove 2
    if time.time() < deadline - 60:
        pts_base = _generate_grid(14, 4)
        score, pts_best = _evaluate_removal(pts_base, 2)
        if pts_best is not None:
            update_best(pts_best, score)
            
    # 3x19 = 57, remove 3 (Not implemented in _evaluate_removal for k=3 yet)
    # Let's add k=3 support or just skip. k=3 is $\binom{57}{3} \approx 29k$ combos. Feasible.
    # Let's update _evaluate_removal to handle k=3?
    # Or just try a few specific removals for k=3?
    # For now, skip k=3 to save time, or implement it.
    
    # 6x10 = 60, remove 6. Too many.
    # 12x5 = 60, remove 6.
    
    # 3. Try grids with additions (extending)
    # 5x10 = 50, add 4.
    # Strategy: Add points to extend rows or columns.
    # e.g., add (1,11), (2,11), (3,11), (4,11) to 5x10?
    # Or add (6,1)..(6,4) to make it 6x10 minus 6?
    # Let's try a few specific addition patterns.
    
    def try_addition(base_a, base_b, add_points):
        base_pts = _generate_grid(base_a, base_b)
        add_arr = np.array(add_points, dtype=np.int64)
        pts = np.vstack([base_pts, add_arr])
        if len(pts) != 54:
            return
        # Check duplicates
        if len(set(map(tuple, pts))) != 54:
            return
        score, _ = _evaluate_removal(pts, 0)
        update_best(pts, score)
        
    if time.time() < deadline - 60:
        # 5x10 + 4 points in a row
        try_addition(5, 10, [(1, 11), (2, 11), (3, 11), (4, 11)])
        try_addition(5, 10, [(6, 1), (6, 2), (6, 3), (6, 4)])
        try_addition(10, 5, [(11, 1), (11, 2), (11, 3), (11, 4)])
        try_addition(10, 5, [(1, 6), (2, 6), (3, 6), (4, 6)])
        
        # 4x12 + 6 points
        # 4x12 = 48. Add 6.
        # Add a row of 6?
        try_addition(4, 12, [(1, 13), (2, 13), (3, 13), (4, 13), (5, 13), (6, 13)])
        try_addition(4, 12, [(5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)])
        
        # 3x18 + 0 (already covered)
        # 2x27 + 0 (already covered)
        
    # Fallback
    if best_points is None:
        pts = _generate_grid(6, 9)
        update_best(pts, _evaluate_removal(pts, 0)[0])
        
    return best_points, best_lines

# EVOLVE-BLOCK-END
