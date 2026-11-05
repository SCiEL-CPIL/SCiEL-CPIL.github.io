import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    def doubled_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    
    def make_grid(w, h):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x - w//2, y - h//2))
        return pts
    
    candidates = [
        make_grid(6, 5),
        make_grid(5, 6),
        make_grid(10, 3),
        make_grid(3, 10),
        make_grid(15, 2),
        make_grid(2, 15)
    ]
    
    best_score = 0
    best_points = candidates[0]
    
    for pts in candidates:
        if len(set(pts)) != 30:
            continue
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            continue
        
        arr = np.array(pts, dtype=np.int64)
        x = arr[:, 0]
        y = arr[:, 1]
        areas = np.abs(x[None, None, :] * (y[None, :, None] - y[:, None, None]) +
                       x[None, :, None] * (y[:, None, None] - y[None, None, :]) +
                       x[:, None, None] * (y[None, None, :] - y[None, :, None]))
        areas_flat = areas.flatten()
        positive = areas_flat[areas_flat > 0]
        if len(positive) == 0:
            continue
        vals, cnts = np.unique(positive, return_counts=True)
        max_c = cnts.max()
        if max_c > best_score:
            best_score = max_c
            best_points = pts[:]
    
    points = best_points[:]
    counts = defaultdict(int)
    for i, j, k in itertools.combinations(range(30), 3):
        d = doubled_area(points[i], points[j], points[k])
        if d > 0:
            counts[d] += 1
    
    # count_of_counts[c] = number of areas d such that counts[d] == c
    count_of_counts = defaultdict(int)
    for c in counts.values():
        count_of_counts[c] += 1
    
    def get_max_count():
        if not count_of_counts:
            return 0
        return max(count_of_counts.keys())
    
    current_max = get_max_count()
    
    # Precompute index pairs for each point i: all (j, k) with j < k, j != i, k != i
    pairs_for_point = []
    for i in range(30):
        others = [j for j in range(30) if j != i]
        pairs = [(j, k) for j_idx, j in enumerate(others) for k in others[j_idx+1:]]
        pairs_for_point.append(pairs)
    
    random.seed(42)
    
    # Best-improvement local search: try all single-point moves, pick the best
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_move = None
        
        for i in range(30):
            if time.time() > deadline - 5:
                break
            old_p = points[i]
            ox, oy = old_p
            
            # Generate candidate positions: nearby + some random
            candidates_pos = set()
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates_pos.add((nx, ny))
            # Add some random positions
            for _ in range(20):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                candidates_pos.add((nx, ny))
            
            for new_p in candidates_pos:
                if new_p == old_p:
                    continue
                if any(p == new_p for p in points if p != old_p):
                    continue
                
                # Evaluate this move
                new_x, new_y = new_p
                old_x, old_y = ox, oy
                
                delta_counts = defaultdict(int)
                for j, k in pairs_for_point[i]:
                    pj, pk = points[j], points[k]
                    old_d = abs(old_x*(pj[1] - pk[1]) + pj[0]*(pk[1] - old_y) + pk[0]*(old_y - pj[1]))
                    new_d = abs(new_x*(pj[1] - pk[1]) + pj[0]*(pk[1] - new_y) + pk[0]*(new_y - pj[1]))
                    if old_d != new_d:
                        if old_d > 0:
                            delta_counts[old_d] -= 1
                        if new_d > 0:
                            delta_counts[new_d] += 1
                
                if not delta_counts:
                    continue
                
                # Apply delta to count_of_counts
                for area, delta in delta_counts.items():
                    old_c = counts.get(area, 0)
                    new_c = old_c + delta
                    if old_c > 0:
                        count_of_counts[old_c] -= 1
                        if count_of_counts[old_c] == 0:
                            del count_of_counts[old_c]
                    if new_c > 0:
                        count_of_counts[new_c] = count_of_counts.get(new_c, 0) + 1
                    counts[area] = new_c
                    if new_c == 0:
                        del counts[area]
                
                new_max = get_max_count()
                
                # Revert count_of_counts and counts
                for area, delta in delta_counts.items():
                    new_c = counts.get(area, 0)
                    old_c = new_c - delta
                    if new_c > 0:
                        count_of_counts[new_c] -= 1
                        if count_of_counts[new_c] == 0:
                            del count_of_counts[new_c]
                    if old_c > 0:
                        count_of_counts[old_c] = count_of_counts.get(old_c, 0) + 1
                    counts[area] = old_c
                    if old_c == 0:
                        del counts[area]
                
                gain = new_max - current_max
                if gain > best_gain:
                    best_gain = gain
                    best_move = (i, new_p, delta_counts)
        
        if best_move is not None and best_gain > 0:
            i, new_p, delta_counts = best_move
            points[i] = new_p
            for area, delta in delta_counts.items():
                old_c = counts.get(area, 0)
                new_c = old_c + delta
                if old_c > 0:
                    count_of_counts[old_c] -= 1
                    if count_of_counts[old_c] == 0:
                        del count_of_counts[old_c]
                if new_c > 0:
                    count_of_counts[new_c] = count_of_counts.get(new_c, 0) + 1
                counts[area] = new_c
                if new_c == 0:
                    del counts[area]
            current_max += best_gain
            best_score = current_max
            best_points = points[:]
            improved = True
    
    return best_points
