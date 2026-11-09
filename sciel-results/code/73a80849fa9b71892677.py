import numpy as np
from itertools import combinations
from math import gcd
import time

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g != 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def evaluate_grid(W, H, num_remove):
    total_points = W * H
    if total_points < 54:
        return -1, None, None
        
    points_full = []
    for y in range(H):
        for x in range(W):
            points_full.append((x, y))
            
    line_dict = {}
    for i in range(total_points):
        p1 = points_full[i]
        for j in range(i + 1, total_points):
            p2 = points_full[j]
            x1, y1 = p1
            x2, y2 = p2
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            norm = normalize_line(A, B, C)
            if norm is None:
                continue
            if norm not in line_dict:
                line_dict[norm] = set()
            line_dict[norm].add(i)
            line_dict[norm].add(j)
            
    line_info = []
    for coeffs, pts_set in line_dict.items():
        A, B, C = coeffs
        mask = 0
        for idx, (x, y) in enumerate(points_full):
            if A * x + B * y + C == 0:
                mask |= (1 << idx)
        if mask.bit_count() >= 2:
            line_info.append((coeffs, mask))
            
    best_score = -1
    best_points = None
    best_lines = None
    full_mask = (1 << total_points) - 1
    indices = list(range(total_points))
    
    if num_remove == 0:
        active_mask = full_mask
        candidates = []
        for coeffs, line_mask in line_info:
            count = (active_mask & line_mask).bit_count()
            if count >= 2:
                candidates.append((count, coeffs))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if len(candidates) >= 27:
            top_27 = candidates[:27]
            score = sum(c for c, _ in top_27)
            if score > best_score:
                best_score = score
                active_indices = [i for i in range(total_points) if (active_mask >> i) & 1]
                pts_out = [points_full[i] for i in active_indices]
                best_points = np.array(pts_out, dtype=np.int64)
                best_lines = np.array([c for _, c in top_27], dtype=np.int64)
                
    elif num_remove == 1:
        for r1 in indices:
            removal_mask = (1 << r1)
            active_mask = full_mask & ~removal_mask
            candidates = []
            for coeffs, line_mask in line_info:
                count = (active_mask & line_mask).bit_count()
                if count >= 2:
                    candidates.append((count, coeffs))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) >= 27:
                top_27 = candidates[:27]
                score = sum(c for c, _ in top_27)
                if score > best_score:
                    best_score = score
                    active_indices = [i for i in range(total_points) if (active_mask >> i) & 1]
                    pts_out = [points_full[i] for i in active_indices]
                    best_points = np.array(pts_out, dtype=np.int64)
                    best_lines = np.array([c for _, c in top_27], dtype=np.int64)
                    
    elif num_remove == 2:
        for r1, r2 in combinations(indices, 2):
            removal_mask = (1 << r1) | (1 << r2)
            active_mask = full_mask & ~removal_mask
            candidates = []
            for coeffs, line_mask in line_info:
                count = (active_mask & line_mask).bit_count()
                if count >= 2:
                    candidates.append((count, coeffs))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) >= 27:
                top_27 = candidates[:27]
                score = sum(c for c, _ in top_27)
                if score > best_score:
                    best_score = score
                    active_indices = [i for i in range(total_points) if (active_mask >> i) & 1]
                    pts_out = [points_full[i] for i in active_indices]
                    best_points = np.array(pts_out, dtype=np.int64)
                    best_lines = np.array([c for _, c in top_27], dtype=np.int64)
                    
    return best_score, best_points, best_lines

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    global_best_score = -1
    global_best_points = None
    global_best_lines = None
    
    # Try different grid configurations
    configs = [
        (7, 8, 2),
        (6, 9, 0),
        (5, 11, 1),
        (4, 14, 2),
        (3, 18, 0),
        (2, 27, 0)
    ]
    
    for W, H, rem in configs:
        if time.time() > deadline - 60:
            break
            
        score, pts, lines = evaluate_grid(W, H, rem)
        if score > global_best_score:
            global_best_score = score
            global_best_points = pts
            global_best_lines = lines
            
    return global_best_points, global_best_lines
