import numpy as np
import time
import random
from math import gcd
from collections import defaultdict

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in (a, b, c):
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _count_incidences(points, lines):
    """Count exact incidences between points (54,2) and lines (27,3)."""
    I = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A * x + B * y + C == 0:
                I += 1
    return I

def _generate_line_pool():
    """Generate a pool of candidate lines with small coefficients."""
    pool = set()
    max_coeff = 4
    max_c = 60
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_c, max_c + 1):
                canon = _normalize_line(A, B, C)
                if canon:
                    pool.add(canon)
    return list(pool)

def _make_grid(rows, cols, offset_x=0, offset_y=0):
    pts = []
    for x in range(offset_x, offset_x + cols):
        for y in range(offset_y, offset_y + rows):
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _make_random(n, seed=42, rng=None):
    if rng is None:
        rng = random.Random(seed)
    pts = []
    seen = set()
    while len(pts) < n:
        x = rng.randint(-8, 15)
        y = rng.randint(-8, 15)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def _local_search(points, pool, point_line_map, deadline, max_iters=500000):
    """
    Local search on point positions. Lines are chosen as top-27 from pool
    based on incidence counts.
    """
    n_pts = len(points)
    n_pool = len(pool)
    
    # Precompute line coefficients as arrays for vectorized evaluation
    A_pool = np.array([l[0] for l in pool], dtype=np.int64)
    B_pool = np.array([l[1] for l in pool], dtype=np.int64)
    C_pool = np.array([l[2] for l in pool], dtype=np.int64)
    
    pts = points.copy()
    
    # Build count vector: counts[j] = number of points on line j
    counts = np.zeros(n_pool, dtype=np.int64)
    for i in range(n_pts):
        x, y = pts[i]
        indices = point_line_map.get((int(x), int(y)), np.array([], dtype=int))
        if len(indices) > 0:
            counts[indices] += 1
    
    def get_top_k_sum(c, k=27):
        if n_pool < k:
            return int(np.sum(c))
        # Use partition for speed
        top_k = np.partition(c, -k)[-k:]
        return int(np.sum(top_k))
    
    current_score = get_top_k_sum(counts)
    best_score = current_score
    best_pts = pts.copy()
    best_counts = counts.copy()
    
    rng = random.Random(42)
    T = 1.0
    T_min = 0.005
    no_improve = 0
    max_no_improve = 2000
    
    start = time.time()
    
    while time.time() < deadline and no_improve < max_no_improve:
        i = rng.randint(0, n_pts - 1)
        old_x, old_y = int(pts[i, 0]), int(pts[i, 1])
        
        # Propose new position
        if rng.random() < 0.85:
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            new_x, new_y = old_x + dx, old_y + dy
        else:
            new_x = rng.randint(-10, 18)
            new_y = rng.randint(-10, 18)
        
        # Check if in map range
        if new_x < -10 or new_x > 18 or new_y < -10 or new_y > 18:
            continue
        
        # Check distinctness (only check against all other points)
        is_dup = False
        for k in range(n_pts):
            if k != i and pts[k, 0] == new_x and pts[k, 1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
        
        # Update counts incrementally
        old_indices = point_line_map.get((old_x, old_y), np.array([], dtype=int))
        new_indices = point_line_map.get((new_x, new_y), np.array([], dtype=int))
        
        old_len = len(old_indices)
        new_len = len(new_indices)
        
        if old_len > 0:
            counts[old_indices] -= 1
        if new_len > 0:
            counts[new_indices] += 1
            
        new_score = get_top_k_sum(counts)
        delta = new_score - current_score
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            pts[i] = [new_x, new_y]
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_counts = counts.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            T *= 0.9995
            if T < T_min:
                T = T_min
        else:
            # Revert
            if old_len > 0:
                counts[old_indices] += 1
            if new_len > 0:
                counts[new_indices] -= 1
            no_improve += 1
    
    # Extract best lines
    top_indices = np.argsort(best_counts)[-27:]
    best_lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
    
    return best_pts, best_lines, best_score

def _polish_with_line_selection(points, deadline):
    """
    Given fixed points, try to find better lines by considering lines 
    defined by pairs of points.
    """
    n_pts = len(points)
    
    # Generate all lines through pairs of points
    candidate_lines = set()
    
    # Add lines with small coefficients
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-60, 61):
                canon = _normalize_line(A, B, C)
                if canon:
                    candidate_lines.add(canon)
    
    # Add lines through pairs of points (if they are integer lines)
    for i in range(n_pts):
        x1, y1 = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n_pts):
            x2, y2 = int(points[j, 0]), int(points[j, 1])
            # Line through (x1,y1) and (x2,y2):
            # (y2-y1)(x-x1) - (x2-x1)(y-y1) = 0
            # (y2-y1)x - (x2-x1)y + (x2-x1)*y1 - (y2-y1)*x1 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x1
            canon = _normalize_line(A, B, C)
            if canon:
                candidate_lines.add(canon)
    
    pool = list(candidate_lines)
    n_pool = len(pool)
    
    A_pool = np.array([l[0] for l in pool], dtype=np.int64)
    B_pool = np.array([l[1] for l in pool], dtype=np.int64)
    C_pool = np.array([l[2] for l in pool], dtype=np.int64)
    
    # Count incidences for each line
    counts = np.zeros(n_pool, dtype=np.int64)
    for i in range(n_pts):
        x, y = int(points[i, 0]), int(points[i, 1])
        vals = A_pool * x + B_pool * y + C_pool
        mask = vals == 0
        counts[mask] += 1
    
    # Select top 27 lines
    top_indices = np.argsort(counts)[-27:]
    lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
    
    # Verify score
    score = _count_incidences(points, lines)
    
    return points, lines, score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin
    
    pool = _generate_line_pool()
    n_pool = len(pool)
    
    # Precompute point_line_map for the bounding box
    A_pool = np.array([l[0] for l in pool], dtype=np.int64)
    B_pool = np.array([l[1] for l in pool], dtype=np.int64)
    C_pool = np.array([l[2] for l in pool], dtype=np.int64)
    
    point_line_map = {}
    for x in range(-10, 19):
        for y in range(-10, 19):
            vals = A_pool * x + B_pool * y + C_pool
            mask = vals == 0
            indices = np.where(mask)[0]
            point_line_map[(x, y)] = indices
    
    # Multiple initial configurations
    configs = [
        _make_grid(9, 6),        # 6x9 grid
        _make_grid(18, 3),       # 3x18 grid
        _make_grid(27, 2),       # 2x27 grid
        _make_grid(9, 6, offset_x=1, offset_y=1),  # shifted grid
        _make_random(54, seed=42),
        _make_random(54, seed=123),
        _make_random(54, seed=777),
    ]
    
    best_overall_pts = None
    best_overall_lines = None
    best_overall_score = -1
    
    # Phase 1: Local search on each config
    time_per_config = (deadline - start_time - 60) / len(configs)  # reserve 60s for polish
    
    for i, init_pts in enumerate(configs):
        t_end = start_time + time_per_config * (i + 1)
        if i == len(configs) - 1:
            t_end = deadline - 60  # reserve time for polish
        
        if time.time() > t_end:
            continue
            
        try:
            pts, lines, score = _local_search(init_pts, pool, point_line_map, t_end)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_pts = pts
                best_overall_lines = lines
        except Exception:
            pass
    
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = _make_grid(9, 6)
        counts = np.zeros(n_pool, dtype=np.int64)
        for i in range(54):
            x, y = best_overall_pts[i]
            indices = point_line_map.get((int(x), int(y)), np.array([], dtype=int))
            if len(indices) > 0:
                counts[indices] += 1
        top_indices = np.argsort(counts)[-27:]
        best_overall_lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
        best_overall_score = _count_incidences(best_overall_pts, best_overall_lines)
    
    # Phase 2: Polish with line selection
    if time.time() < deadline - 5:
        try:
            polished_pts, polished_lines, polished_score = _polish_with_line_selection(
                best_overall_pts, deadline - 5
            )
            if polished_score > best_overall_score:
                best_overall_score = polished_score
                best_overall_pts = polished_pts
                best_overall_lines = polished_lines
        except Exception:
            pass
    
    # Phase 3: Another round of local search with the best lines as guide
    if time.time() < deadline - 10 and best_overall_score > 0:
        # Try to improve by moving points to be on the selected lines
        # This is a heuristic: for each point, try positions that are on one of the selected lines
        try:
            selected_lines = best_overall_lines
            n_selected = len(selected_lines)
            
            # For each selected line, generate candidate points on that line
            candidate_positions = set()
            for line in selected_lines:
                A, B, C = int(line[0]), int(line[1]), int(line[2])
                if A == 0:
                    # B*y + C = 0 => y = -C/B
                    if B != 0 and (-C) % B == 0:
                        y_val = -C // B
                        for x in range(-10, 19):
                            candidate_positions.add((x, y_val))
                elif B == 0:
                    # A*x + C = 0 => x = -C/A
                    if A != 0 and (-C) % A == 0:
                        x_val = -C // A
                        for y in range(-10, 19):
                            candidate_positions.add((x_val, y))
                else:
                    # A*x + B*y + C = 0
                    # Try all x in range, check if y is integer
                    for x in range(-10, 19):
                        num = -A * x - C
                        if num % B == 0:
                            y_val = num // B
                            if -10 <= y_val <= 18:
                                candidate_positions.add((x, y_val))
            
            # Now do a quick local search where moves are biased toward candidate positions
            if candidate_positions:
                cand_list = list(candidate_positions)
                pts = best_overall_pts.copy()
                
                # Rebuild counts for current configuration
                counts = np.zeros(n_pool, dtype=np.int64)
                for i in range(54):
                    x, y = int(pts[i, 0]), int(pts[i, 1])
                    indices = point_line_map.get((x, y), np.array([], dtype=int))
                    if len(indices) > 0:
                        counts[indices] += 1
                
                def get_top_k_sum(c, k=27):
                    if n_pool < k:
                        return int(np.sum(c))
                    top_k = np.partition(c, -k)[-k:]
                    return int(np.sum(top_k))
                
                current_score = get_top_k_sum(counts)
                best_score = current_score
                best_pts_phase3 = pts.copy()
                
                rng = random.Random(999)
                start_p3 = time.time()
                end_p3 = deadline - 5
                
                no_improve = 0
                max_no_improve = 1000
                
                while time.time() < end_p3 and no_improve < max_no_improve:
                    i = rng.randint(0, 53)
                    old_x, old_y = int(pts[i, 0]), int(pts[i, 1])
                    
                    # 70% chance to move to a candidate position, 30% random
                    if rng.random() < 0.7 and len(cand_list) > 0:
                        new_x, new_y = cand_list[rng.randint(0, len(cand_list) - 1)]
                    else:
                        new_x = old_x + rng.randint(-2, 2)
                        new_y = old_y + rng.randint(-2, 2)
                    
                    if new_x < -10 or new_x > 18 or new_y < -10 or new_y > 18:
                        continue
                    
                    is_dup = False
                    for k in range(54):
                        if k != i and pts[k, 0] == new_x and pts[k, 1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    old_indices = point_line_map.get((old_x, old_y), np.array([], dtype=int))
                    new_indices = point_line_map.get((new_x, new_y), np.array([], dtype=int))
                    
                    old_len = len(old_indices)
                    new_len = len(new_indices)
                    
                    if old_len > 0:
                        counts[old_indices] -= 1
                    if new_len > 0:
                        counts[new_indices] += 1
                        
                    new_score = get_top_k_sum(counts)
                    delta = new_score - current_score
                    
                    if delta >= 0 or rng.random() < np.exp(delta / 0.5):
                        pts[i] = [new_x, new_y]
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_pts_phase3 = pts.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        if old_len > 0:
                            counts[old_indices] += 1
                        if new_len > 0:
                            counts[new_indices] -= 1
                        no_improve += 1
                
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_pts = best_pts_phase3
                    # Re-select lines
                    top_indices = np.argsort(counts)[-27:]
                    best_overall_lines = np.array([pool[i] for i in top_indices], dtype=np.int64)
        except Exception:
            pass
    
    return best_overall_pts, best_overall_lines
