import numpy as np
import time
from math import gcd
from itertools import combinations
import random

def _normalize_line(a, b, c):
    a, b, c = int(a), int(b), int(c)
    if a == 0 and b == 0:
        return None
    g = gcd(abs(a), gcd(abs(b), abs(c)))
    if g == 0:
        return None
    a, b, c = a // g, b // g, c // g
    for v in [a, b, c]:
        if v != 0:
            if v < 0:
                a, b, c = -a, -b, -c
            break
    return (a, b, c)

def _is_valid(points, lines):
    try:
        p = np.asarray(points)
        l = np.asarray(lines)
    except Exception:
        return False
    if p.shape != (54, 2) or l.shape != (27, 3):
        return False
    for arr in [p, l]:
        if not np.all(np.isfinite(arr)):
            return False
        if not np.all(np.abs(arr - np.round(arr)) < 1e-6):
            return False
    if len(set(map(tuple, np.round(p, 6).astype(int).tolist()))) != 54:
        return False
    seen_lines = set()
    for row in np.round(l, 6).astype(int).tolist():
        canon = _normalize_line(row[0], row[1], row[2])
        if canon is None:
            return False
        if canon in seen_lines:
            return False
        seen_lines.add(canon)
    return True

def _compute_incidence_matrix(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # vals[i, j] = A_i * x_j + B_i * y_j + C_i
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return (vals == 0).astype(np.int64)

def _incidence_count(points, lines):
    return int(np.sum(_compute_incidence_matrix(points, lines)))

def _initial_grid_diagonal():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
    
    pts = np.array(points, dtype=np.int64)
    
    # Start with all vertical and horizontal lines
    lines = []
    for x in range(w):
        lines.append((1, 0, -x))
    for y in range(h):
        lines.append((0, 1, -y))
    
    # Generate all secant lines through pairs of points
    secants = set()
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            c = _normalize_line(A, B, C)
            if c:
                secants.add(c)
    
    # Score all candidate lines (secants + existing)
    all_cands = list(set(lines + list(secants)))
    scored = []
    for l in all_cands:
        A, B, C = l
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(vals == 0))
        scored.append((cnt, l))
    
    scored.sort(key=lambda x: -x[0])
    
    # Select top 27 unique lines
    current_lines = set()
    final_lines = []
    for cnt, l in scored:
        canon = _normalize_line(*l)
        if canon not in current_lines:
            final_lines.append(canon)
            current_lines.add(canon)
            if len(final_lines) == 27:
                break
    
    return np.array(points, dtype=np.int64), np.array(final_lines, dtype=np.int64)

def _local_search(points, lines, deadline):
    import random
    rng = random.Random(123)
    
    pts = points.copy()
    lns = lines.copy()
    n_pts, n_lines = 54, 27
    
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    best_inc = inc_matrix.copy()
    
    # Generate candidate lines: secants of current points + grid lines
    cand_lines = set()
    # Add grid-like lines
    for A in range(-5, 6):
        for B in range(-5, 6):
            if A == 0 and B == 0: continue
            for C in range(-15, 16):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
                
    # Secants through all pairs
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
            
    cand_lines_list = list(cand_lines)
    
    def get_cand_points():
        cands = set()
        for p in pts:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    cands.add((int(p[0])+dx, int(p[1])+dy))
        return list(cands)
    
    T = 8.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 5:
        r = rng.random()
        if r < 0.45:
            # Move point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            cands = get_cand_points()
            new_pos = cands[rng.randint(0, len(cands)-1)]
            
            if new_pos == old_pos:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    dup = True
                    break
            if dup:
                continue
            
            # Vectorized delta
            A = lns[:, 0]
            B = lns[:, 1]
            C = lns[:, 2]
            new_vals = A * new_pos[0] + B * new_pos[1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            
            old_inc = inc_matrix[:, i]
            
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pts[i] = new_pos
                inc_matrix[:, i] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    best_inc = inc_matrix.copy()
        elif r < 0.90:
            # Move line
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines_list[rng.randint(0, len(cand_lines_list)-1)]
            
            if new_line == old_canon:
                continue
            
            # Check duplicates
            dup = False
            for k in range(n_lines):
                if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                    dup = True
                    break
            if dup:
                continue
            
            # Vectorized delta
            A, B, C = new_line
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            new_inc = (new_vals == 0).astype(np.int64)
            
            old_inc = inc_matrix[j, :]
            
            delta = int(np.sum(new_inc)) - int(np.sum(old_inc))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                inc_matrix[j, :] = new_inc
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    best_inc = inc_matrix.copy()
        else:
            # Swap two points
            i, k = rng.sample(range(n_pts), 2)
            pos_i, pos_k = tuple(pts[i]), tuple(pts[k])
            
            # Check if swap creates duplicate
            if pos_i == pos_k:
                continue
            
            # Compute delta for swapping
            A = lns[:, 0]
            B = lns[:, 1]
            C = lns[:, 2]
            
            new_vals_i = A * pos_k[0] + B * pos_k[1] + C
            new_inc_i = (new_vals_i == 0).astype(np.int64)
            
            new_vals_k = A * pos_i[0] + B * pos_i[1] + C
            new_inc_k = (new_vals_k == 0).astype(np.int64)
            
            old_inc_i = inc_matrix[:, i]
            old_inc_k = inc_matrix[:, k]
            
            delta = (int(np.sum(new_inc_i)) + int(np.sum(new_inc_k))) - (int(np.sum(old_inc_i)) + int(np.sum(old_inc_k)))
            
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pts[i] = pos_k
                pts[k] = pos_i
                inc_matrix[:, i] = new_inc_i
                inc_matrix[:, k] = new_inc_k
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    best_inc = inc_matrix.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_pts, best_lns

def _refine_points(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_pts):
            best_pos = tuple(pts[i])
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[:, i]))
            
            # Try nearby positions with larger radius
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    new_pos = (int(pts[i, 0]) + dx, int(pts[i, 1]) + dy)
                    if new_pos == best_pos:
                        continue
                    dup = False
                    for k in range(n_pts):
                        if k != i and tuple(pts[k]) == new_pos:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    A = lns[:, 0]
                    B = lns[:, 1]
                    C = lns[:, 2]
                    new_vals = A * new_pos[0] + B * new_pos[1] + C
                    new_inc = (new_vals == 0).astype(np.int64)
                    new_sum = int(np.sum(new_inc))
                    delta = new_sum - old_sum
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_pos = new_pos
                        best_new_inc = new_inc
            
            if best_delta > 0:
                pts[i] = best_pos
                inc_matrix[:, i] = best_new_inc
                current_score += best_delta
                improved = True
    
    return pts

def _refine_lines(pts, lns, deadline):
    n_pts, n_lines = 54, 27
    inc_matrix = _compute_incidence_matrix(pts, lns)
    current_score = int(np.sum(inc_matrix))
    improved = True
    
    # Pre-generate candidate lines
    cand_lines = set()
    for A in range(-8, 9):
        for B in range(-8, 9):
            if A == 0 and B == 0: continue
            for C in range(-25, 26):
                c = _normalize_line(A, B, C)
                if c: cand_lines.add(c)
    # Secants
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            p1 = pts[i]
            p2 = pts[j]
            A = p2[1] - p1[1]
            B = p1[0] - p2[0]
            C = -(A * p1[0] + B * p1[1])
            c = _normalize_line(A, B, C)
            if c: cand_lines.add(c)
            
    cand_lines_list = list(cand_lines)
    
    # Pre-compute incidence counts for all candidate lines
    cand_inc_counts = {}
    for cl in cand_lines_list:
        A, B, C = cl
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cand_inc_counts[cl] = int(np.sum(vals == 0))
    
    while improved and time.time() < deadline - 10:
        improved = False
        for j in range(n_lines):
            best_line = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            best_delta = 0
            best_new_inc = None
            old_sum = int(np.sum(inc_matrix[j, :]))
            
            for new_line in cand_lines_list:
                if new_line == best_line:
                    continue
                dup = False
                for k in range(n_lines):
                    if k != j and _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2])) == new_line:
                        dup = True
                        break
                if dup:
                    continue
                
                new_sum = cand_inc_counts[new_line]
                delta = new_sum - old_sum
                
                if delta > best_delta:
                    best_delta = delta
                    best_line = new_line
                    A, B, C = new_line
                    best_new_inc = (A * pts[:, 0] + B * pts[:, 1] + C == 0).astype(np.int64)
            
            if best_delta > 0:
                lns[j] = best_line
                inc_matrix[j, :] = best_new_inc
                current_score += best_delta
                improved = True
    
    return lns

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 595 # Safety margin
    
    # 1. Initial configuration
    pts, lns = _initial_grid_diagonal()
    
    # 2. Refine points
    if time.time() < deadline - 80:
        pts = _refine_points(pts, lns, deadline - 80)
    
    # 3. Refine lines
    if time.time() < deadline - 80:
        lns = _refine_lines(pts, lns, deadline - 80)
    
    # 4. Local search with simulated annealing (main search)
    if time.time() < deadline - 30:
        pts, lns = _local_search(pts, lns, deadline - 15)
    
    # 5. Final refinement
    if time.time() < deadline - 5:
        pts = _refine_points(pts, lns, deadline - 5)
        lns = _refine_lines(pts, lns, deadline - 2)
    
    # 6. Validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = []
        for x in range(6):
            lines.append((1, 0, -x))
        for y in range(9):
            lines.append((0, 1, -y))
        for c in range(0, 12):
            lines.append((1, 1, -c))
        lns = np.array(lines, dtype=np.int64)
    
    return pts, lns
