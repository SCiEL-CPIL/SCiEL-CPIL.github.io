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
    """Joint greedy construction: alternate between selecting best lines and placing best points."""
    n_pts = 54
    n_lines = 27
    
    # Start with a 6x9 grid
    w, h = 6, 9
    points = [(x, y) for x in range(w) for y in range(h)]
    
    # Build a large candidate line pool
    cand_lines = set()
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-25, 26):
                c = _normalize_line(A, B, C)
                if c:
                    cand_lines.add(c)
    cand_lines = list(cand_lines)
    
    # Candidate point positions (small grid around origin)
    cand_points = set()
    for x in range(-5, 10):
        for y in range(-5, 15):
            cand_points.add((x, y))
    cand_points = list(cand_points)
    
    pts = np.array(points, dtype=np.int64)
    lns = np.zeros((n_lines, 3), dtype=np.int64)
    active_lines = [False] * n_lines
    
    # Phase 1: Greedily select lines
    for j in range(n_lines):
        best_cnt = -1
        best_line = None
        for cl in cand_lines:
            A, B, C = cl
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            cnt = int(np.sum(vals == 0))
            if cnt > best_cnt:
                best_cnt = cnt
                best_line = cl
        if best_line is not None:
            lns[j] = best_line
            active_lines[j] = True
    
    # Phase 2: Iteratively improve by moving points and replacing lines
    for iteration in range(200):
        improved = False
        
        # Try moving each point to a better position
        for i in range(n_pts):
            old_pos = tuple(pts[i])
            best_delta = 0
            best_new_pos = None
            
            # Try nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                    
                    # Check duplicates
                    is_dup = False
                    for k in range(n_pts):
                        if k != i and tuple(pts[k]) == new_pos:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    
                    # Calculate delta
                    delta = 0
                    for j in range(n_lines):
                        if not active_lines[j]:
                            continue
                        A, B, C = lns[j]
                        was = (A * old_pos[0] + B * old_pos[1] + C == 0)
                        now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                        delta += (1 if now else 0) - (1 if was else 0)
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pos = new_pos
            
            if best_new_pos is not None and best_delta > 0:
                pts[i] = best_new_pos
                improved = True
        
        # Try replacing each line
        for j in range(n_lines):
            if not active_lines[j]:
                continue
            old_line = tuple(lns[j])
            best_delta = 0
            best_new_line = None
            
            for cl in cand_lines:
                if cl == old_line:
                    continue
                A, B, C = cl
                old_A, old_B, old_C = old_line
                delta = 0
                for i in range(n_pts):
                    x, y = pts[i]
                    was = (old_A * x + old_B * y + old_C == 0)
                    now = (A * x + B * y + C == 0)
                    delta += (1 if now else 0) - (1 if was else 0)
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_line = cl
            
            if best_new_line is not None and best_delta > 0:
                lns[j] = best_new_line
                improved = True
        
        if not improved:
            break
    
    return pts, lns

def _local_search(pts, lns, deadline):
    rng = random.Random(42)
    pts = pts.copy()
    lns = lns.copy()
    
    current_score = _incidence_count(pts, lns)
    best_score = current_score
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    n_pts = 54
    n_lines = 27
    
    # Precompute candidate lines pool for random replacement
    # Small coefficients
    cand_pool = set()
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-25, 26):
                c = _normalize_line(A, B, C)
                if c:
                    cand_pool.add(c)
    cand_pool = list(cand_pool)
    
    T_start = 10.0
    T_end = 0.1
    max_iters = 500000 # Estimate safe number
    iters_done = 0
    
    while time.time() < deadline - 10:
        iters_done += 1
        # Cool down
        progress = min(iters_done / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        if rng.random() < 0.6:
            # Move a point
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            
            # Try small moves
            moved = False
            for _ in range(15):
                dx = rng.randint(-3, 3)
                dy = rng.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                
                # Check duplicates
                is_dup = False
                for k in range(n_pts):
                    if k != i and tuple(pts[k]) == new_pos:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                # Calculate delta
                delta = 0
                for j in range(n_lines):
                    A, B, C = lns[j]
                    was = (A * old_pos[0] + B * old_pos[1] + C == 0)
                    now = (A * new_pos[0] + B * new_pos[1] + C == 0)
                    delta += (1 if now else 0) - (1 if was else 0)
                    
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                    pts[i] = new_pos
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
            
            # Generate candidate: either from pool or defined by 2 random points
            if rng.random() < 0.5:
                new_line = cand_pool[rng.randint(0, len(cand_pool)-1)]
            else:
                # Pick 2 distinct points
                i1 = rng.randint(0, n_pts - 1)
                i2 = rng.randint(0, n_pts - 1)
                if i1 == i2:
                    continue
                p1 = pts[i1]
                p2 = pts[i2]
                x1, y1 = p1
                x2, y2 = p2
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                new_line = _normalize_line(A, B, C)
                if new_line is None:
                    continue
            
            if new_line == old_line:
                continue
                
            # Check duplicates
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    if tuple(lns[k]) == new_line:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            # Calculate delta
            delta = 0
            A, B, C = new_line
            old_A, old_B, old_C = old_line
            for i in range(n_pts):
                x, y = pts[i]
                was = (old_A * x + old_B * y + old_C == 0)
                now = (A * x + B * y + C == 0)
                delta += (1 if now else 0) - (1 if was else 0)
                
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                lns[j] = new_line
                current_score += delta
                
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                best_lns = lns.copy()
                
    return best_pts, best_lns, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    # 1. Initialize with joint greedy
    pts, lns = _generate_grid_initialization()
    
    # 2. Local Search
    pts, lns, score = _local_search(pts, lns, deadline)
    
    # 3. Final Validation and Cleanup
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
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
