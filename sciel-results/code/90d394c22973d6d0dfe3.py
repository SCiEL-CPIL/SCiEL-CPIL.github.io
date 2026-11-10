import numpy as np
import time
from math import gcd
from collections import defaultdict
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

def _incidence_count(points, lines):
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    vals = lns[:, 0, None] * pts[None, :, 0] + lns[:, 1, None] * pts[None, :, 1] + lns[:, 2, None]
    return int(np.sum(vals == 0))

def _generate_grid_initialization():
    w, h = 6, 9
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x, y))
            
    lines = []
    for x in range(w):
        lines.append((1, 0, -x))
    for y in range(h):
        lines.append((0, 1, -y))
    for c in range(-h+1, w):
        y_min = max(0, -c)
        y_max = min(h, w-c)
        if y_max > y_min:
            lines.append((1, -1, -c))
    for c in range(0, w+h-1):
        y_min = max(0, c-w+1)
        y_max = min(h, c+1)
        if y_max > y_min:
            lines.append((1, 1, -c))
            
    pts_arr = np.array(points)
    line_lengths = []
    for l in lines:
        A, B, C = l
        vals = A * pts_arr[:, 0] + B * pts_arr[:, 1] + C
        cnt = int(np.sum(vals == 0))
        line_lengths.append((cnt, l))
        
    line_lengths.sort(key=lambda x: -x[0])
    
    selected_lines = []
    seen = set()
    for cnt, l in line_lengths:
        canon = _normalize_line(l[0], l[1], l[2])
        if canon not in seen:
            seen.add(canon)
            selected_lines.append(canon)
            if len(selected_lines) == 27:
                break
                
    return np.array(points, dtype=np.int64), np.array(selected_lines, dtype=np.int64)

def _build_candidate_pool(max_coord=15, max_coeff=8):
    """Build a large pool of candidate lines with small coefficients."""
    pool = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_coord * max_coeff, max_coord * max_coeff + 1):
                c = _normalize_line(A, B, C)
                if c:
                    pool.add(c)
    return list(pool)

def _soft_top_k_sum(sizes, k, temperature=1.0):
    """Smooth surrogate for sum of top-k values using log-sum-exp trick.
    This approximates sum of top-k when temperature is small."""
    if len(sizes) == 0:
        return 0.0
    s = np.array(sizes, dtype=np.float64)
    # Use the fact that sum of top k ≈ max(0, sum(s - t)) where t is k-th largest
    # Approximate with soft-max weighted sum
    shifted = s - np.max(s)
    weights = np.exp(shifted / temperature)
    weights /= weights.sum()
    # Sort and take top k contribution
    order = np.argsort(s)[::-1]
    top_k = s[order[:k]]
    # Blend: as temperature -> 0, this approaches sum of top k
    return float(np.sum(top_k))

def _dual_phase_select_lines(pts, n_lines=27, max_candidates=500):
    """Dual-space: given points, select the n_lines lines with highest incidence.
    This is the 'hypergraph' view: each line is an edge, we want top-k edges."""
    pts = np.asarray(pts, dtype=np.int64)
    n_pts = len(pts)
    
    # Build candidate lines from pairs of points
    cand_lines = set()
    for i1 in range(n_pts):
        for i2 in range(i1 + 1, n_pts):
            x1, y1 = pts[i1, 0], pts[i1, 1]
            x2, y2 = pts[i2, 0], pts[i2, 1]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            c = _normalize_line(A, B, C)
            if c:
                cand_lines.add(c)
    
    # Also add axis-aligned and simple diagonal lines
    for x in range(-5, 10):
        cand_lines.add(_normalize_line(1, 0, -x))
    for y in range(-5, 10):
        cand_lines.add(_normalize_line(0, 1, -y))
    for c in range(-15, 15):
        cand_lines.add(_normalize_line(1, -1, -c))
        cand_lines.add(_normalize_line(1, 1, -c))
    for c in range(-10, 10):
        cand_lines.add(_normalize_line(2, -1, -c))
        cand_lines.add(_normalize_line(1, -2, -c))
        cand_lines.add(_normalize_line(2, 1, -c))
        cand_lines.add(_normalize_line(1, 2, -c))
    
    cand_lines = [c for c in cand_lines if c is not None]
    
    # Score each candidate line by its incidence count
    line_scores = []
    for c in cand_lines:
        A, B, C = c
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        cnt = int(np.sum(vals == 0))
        line_scores.append((cnt, c))
    
    line_scores.sort(key=lambda x: -x[0])
    
    # Select top n_lines with no duplicates
    selected = []
    seen = set()
    for cnt, c in line_scores:
        if c not in seen:
            seen.add(c)
            selected.append(c)
            if len(selected) == n_lines:
                break
    
    return np.array(selected, dtype=np.int64)

def _dual_phase_select_points(lns, n_pts=54, max_candidates=200):
    """Dual-space: given lines, select n_pts points that maximize total incidence.
    Each point's score is the number of lines it lies on. We want top-k points."""
    lns = np.asarray(lns, dtype=np.int64)
    n_lines = len(lns)
    
    # Build candidate points: intersections of pairs of lines
    cand_pts = set()
    for i1 in range(n_lines):
        for i2 in range(i1 + 1, n_lines):
            A1, B1, C1 = lns[i1]
            A2, B2, C2 = lns[i2]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            x_num = B1 * C2 - B2 * C1
            y_num = A2 * C1 - A1 * C2
            if x_num % det == 0 and y_num % det == 0:
                x = x_num // det
                y = y_num // det
                if abs(x) < 30 and abs(y) < 30:
                    cand_pts.add((x, y))
    
    # Also add grid points in a reasonable range
    for x in range(-8, 12):
        for y in range(-8, 12):
            cand_pts.add((x, y))
    
    cand_pts = list(cand_pts)
    
    # Score each candidate point by its incidence count
    pt_scores = []
    for (x, y) in cand_pts:
        cnt = 0
        for j in range(n_lines):
            A, B, C = lns[j]
            if A * x + B * y + C == 0:
                cnt += 1
        pt_scores.append((cnt, (x, y)))
    
    pt_scores.sort(key=lambda x: -x[0])
    
    # Select top n_pts with no duplicates
    selected = []
    seen = set()
    for cnt, p in pt_scores:
        if p not in seen:
            seen.add(p)
            selected.append(p)
            if len(selected) == n_pts:
                break
    
    return np.array(selected, dtype=np.int64)

def _local_search(pts, lns, deadline, cand_pool):
    rng = random.Random(42)
    pts = pts.copy()
    lns = lns.copy()
    
    current_score = _incidence_count(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    n_pts = 54
    n_lines = 27
    
    def compute_incidence(pts, lns):
        return (lns[:, 0, None] * pts[None, :, 0] + 
                lns[:, 1, None] * pts[None, :, 1] + 
                lns[:, 2, None]) == 0
    
    inc = compute_incidence(pts, lns)
    line_sizes = inc.sum(axis=1)
    
    T_start = 8.0
    T_end = 0.05
    max_iters = 800000
    iters_done = 0
    
    while time.time() < deadline - 15:
        iters_done += 1
        progress = min(iters_done / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Occasionally do a dual-phase update
        if rng.random() < 0.03:
            # Dual: reselect lines given current points
            new_lns = _dual_phase_select_lines(pts)
            if new_lns.shape == (n_lines, 3):
                new_score = _incidence_count(pts, new_lns)
                if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                    lns = new_lns
                    inc = compute_incidence(pts, lns)
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                        best_lns = lns.copy()
            continue
        
        if rng.random() < 0.55:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = (pts[i, 0], pts[i, 1])
            
            moved = False
            for _ in range(15):
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                
                is_dup = False
                for k in range(n_pts):
                    if k != i and pts[k, 0] == new_pos[0] and pts[k, 1] == new_pos[1]:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                delta = 0
                for j in range(n_lines):
                    A, B, C = lns[j]
                    was = inc[j, i]
                    now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                    delta += (1 if now else 0) - (1 if was else 0)
                    
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    pts[i] = new_pos
                    for j in range(n_lines):
                        A, B, C = lns[j]
                        inc[j, i] = (A * new_pos[0] + B * new_pos[1] + C == 0)
                    current_score += delta
                    moved = True
                    break
            
            if not moved:
                continue
            
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
        else:
            # Replace a line
            j = rng.randint(0, n_lines - 1)
            old_line = tuple(lns[j])
            
            if rng.random() < 0.4:
                new_line = cand_pool[rng.randint(0, len(cand_pool)-1)]
            else:
                i1 = rng.randint(0, n_pts - 1)
                i2 = rng.randint(0, n_pts - 1)
                if i1 == i2:
                    continue
                x1, y1 = pts[i1, 0], pts[i1, 1]
                x2, y2 = pts[i2, 0], pts[i2, 1]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                new_line = _normalize_line(A, B, C)
                if new_line is None:
                    continue
            
            if new_line == old_line:
                continue
                
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    if tuple(lns[k]) == new_line:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            delta = 0
            A, B, C = new_line
            for i in range(n_pts):
                x, y = pts[i, 0], pts[i, 1]
                was = inc[j, i]
                now = (A * x + B * y + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                for i in range(n_pts):
                    inc[j, i] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
                current_score += delta
                
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
    return best_pts, best_lns, best_score

def _polish(pts, lns, deadline):
    """Final high-precision polish pass with dual-space refinement."""
    rng = random.Random(123)
    pts = pts.copy()
    lns = lns.copy()
    
    current_score = _incidence_count(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    n_pts = 54
    n_lines = 27
    
    def compute_incidence(pts, lns):
        return (lns[:, 0, None] * pts[None, :, 0] + 
                lns[:, 1, None] * pts[None, :, 1] + 
                lns[:, 2, None]) == 0
    
    inc = compute_incidence(pts, lns)
    
    # Try all single-point moves in a neighborhood
    for i in range(n_pts):
        if time.time() > deadline - 10:
            break
        x0, y0 = pts[i, 0], pts[i, 1]
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                new_pos = (x0 + dx, y0 + dy)
                
                is_dup = False
                for k in range(n_pts):
                    if k != i and pts[k, 0] == new_pos[0] and pts[k, 1] == new_pos[1]:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                delta = 0
                for j in range(n_lines):
                    A, B, C = lns[j]
                    was = inc[j, i]
                    now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                    delta += (1 if now else 0) - (1 if was else 0)
                
                if delta > 0:
                    pts[i] = new_pos
                    for j in range(n_lines):
                        A, B, C = lns[j]
                        inc[j, i] = (A * new_pos[0] + B * new_pos[1] + C == 0)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_pts = pts.copy()
                        best_lns = lns.copy()
    
    # Dual-space: reselect lines given optimized points
    if time.time() < deadline - 8:
        new_lns = _dual_phase_select_lines(pts)
        if new_lns.shape == (n_lines, 3):
            new_score = _incidence_count(pts, new_lns)
            if new_score > current_score:
                lns = new_lns
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
    
    # Dual-space: reselect points given optimized lines
    if time.time() < deadline - 5:
        new_pts = _dual_phase_select_points(lns)
        if new_pts.shape == (n_pts, 2):
            new_score = _incidence_count(new_pts, lns)
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
                    best_lns = lns.copy()
    
    # Final line replacement polish
    extra_pool = set()
    for i1 in range(n_pts):
        for i2 in range(i1+1, n_pts):
            x1, y1 = pts[i1, 0], pts[i1, 1]
            x2, y2 = pts[i2, 0], pts[i2, 1]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            c = _normalize_line(A, B, C)
            if c:
                extra_pool.add(c)
    
    inc = compute_incidence(pts, lns)
    
    for j in range(n_lines):
        if time.time() > deadline - 3:
            break
        best_delta = 0
        best_new_line = None
        
        for cand in extra_pool:
            if cand == tuple(lns[j]):
                continue
            is_dup = False
            for k in range(n_lines):
                if k != j and tuple(lns[k]) == cand:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            A, B, C = cand
            delta = 0
            for i in range(n_pts):
                x, y = pts[i, 0], pts[i, 1]
                was = inc[j, i]
                now = (A * x + B * y + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
            
            if delta > best_delta:
                best_delta = delta
                best_new_line = cand
        
        if best_new_line is not None:
            lns[j] = best_new_line
            A, B, C = best_new_line
            for i in range(n_pts):
                inc[j, i] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
            current_score += best_delta
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
    
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 570
    
    # 1. Initialize
    pts, lns = _generate_grid_initialization()
    
    # 2. Build large candidate pool
    cand_pool = _build_candidate_pool(max_coord=15, max_coeff=8)
    
    # 3. Dual-phase: reselect lines for initial points
    if time.time() < deadline - 400:
        lns = _dual_phase_select_lines(pts)
    
    # 4. Local Search
    pts, lns, score = _local_search(pts, lns, deadline, cand_pool)
    
    # 5. Polish pass
    if time.time() < deadline - 20:
        pts, lns, score = _polish(pts, lns, deadline)
    
    # 6. Final dual-phase refinement
    if time.time() < deadline - 5:
        lns = _dual_phase_select_lines(pts)
        new_score = _incidence_count(pts, lns)
        if new_score > score:
            score = new_score
        pts2 = _dual_phase_select_points(lns)
        new_score2 = _incidence_count(pts2, lns)
        if new_score2 > score:
            pts = pts2
            score = new_score2
    
    # 7. Final Validation and Cleanup
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
        for k in range(12):
            lines.append((1, 1, -k))
        lns = np.array(lines, dtype=np.int64)
        
    return pts, lns
