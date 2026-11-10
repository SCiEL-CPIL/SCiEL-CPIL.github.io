import numpy as np
import time
import random
from math import gcd
from collections import defaultdict
from itertools import combinations

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
    vals = lns[:, 0, None, None] * pts[:, 0, None] + lns[:, 1, None, None] * pts[:, 1, None] + lns[:, 2, None, None]
    return int(np.sum(vals == 0))

def _line_intersection(l1, l2):
    A1, B1, C1 = l1
    A2, B2, C2 = l2
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x_num = B1 * C2 - B2 * C1
    y_num = A2 * C1 - A1 * C2
    if x_num % det != 0 or y_num % det != 0:
        return None
    x = x_num // det
    y = y_num // det
    if abs(x) > 200 or abs(y) > 200:
        return None
    return (x, y)

def _generate_lattice_lines_through_points(points, max_coeff=12, max_const=60):
    """
    Given a set of points, generate all lines that pass through at least 2 of these points.
    """
    lines = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= max_coeff and abs(B) <= max_coeff and abs(C) <= max_const:
                canon = _normalize_line(A, B, C)
                if canon:
                    lines.add(canon)
    return list(lines)

def _direct_construction_v2():
    """
    Strategy:
    1. Define a set of 54 points on a 6x9 grid.
    2. Find all lines that pass through at least 2 of these points.
    3. Select the top 27 lines (by number of points they pass through).
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    candidate_lines = _generate_lattice_lines_through_points(points, max_coeff=12, max_const=60)
    
    line_scores = {}
    for line in candidate_lines:
        A, B, C = line
        cnt = 0
        for pt in points:
            x, y = pt
            if A * x + B * y + C == 0:
                cnt += 1
        line_scores[line] = cnt
    
    sorted_lines = sorted(line_scores.items(), key=lambda x: -x[1])
    
    selected_lines = []
    seen = set()
    for line, score in sorted_lines:
        if score < 2:
            break
        if line not in seen:
            selected_lines.append(line)
            seen.add(line)
        if len(selected_lines) == 27:
            break
    
    return points, selected_lines

def _local_search(points, lines, deadline):
    """
    Fast local search to improve the direct construction.
    Uses vectorized delta calculations for speed.
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Build set of current points for fast lookup
    current_point_set = set(map(tuple, pts.tolist()))
    
    # Candidate points: intersections of current lines + neighbors
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    
    # Add current points
    for p in current_point_set:
        cand_points.add(p)
    
    # Candidate lines: lines through pairs of current points
    cand_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 15 and abs(B) <= 15 and abs(C) <= 60:
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    
    # Add current lines
    for j in range(n_lines):
        canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        if canon:
            cand_lines.add(canon)
    
    # Add some random lines
    for _ in range(300):
        A = rng.randint(-8, 8)
        B = rng.randint(-8, 8)
        if A == 0 and B == 0:
            continue
        C = rng.randint(-30, 30)
        canon = _normalize_line(A, B, C)
        if canon:
            cand_lines.add(canon)
            
    cand_points = list(cand_points)
    cand_lines = list(cand_lines)
    
    if not cand_points or not cand_lines:
        return best_pts, best_lns, best_total
    
    # Precompute canonical forms for cand_lines
    cand_lines_canon = []
    for cl in cand_lines:
        canon = _normalize_line(cl[0], cl[1], cl[2])
        if canon:
            cand_lines_canon.append(canon)
    cand_lines = cand_lines_canon
    
    # Precompute line coefficients as numpy arrays for faster computation
    lns_A = lns[:, 0].astype(np.int64)
    lns_B = lns[:, 1].astype(np.int64)
    lns_C = lns[:, 2].astype(np.int64)
    
    T_start = 5.0
    T_end = 0.01
    n_iters = 100000
    iter_count = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        if rng.random() < 0.5:
            # Point move
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            if new_pos in current_point_set:
                # Check if it's the same point
                if new_pos == old_pos:
                    continue
                # Find if another point is at this position
                found = False
                for k in range(n_pts):
                    if k != i and tuple(pts[k]) == new_pos:
                        found = True
                        break
                if found:
                    continue
            
            x_new, y_new = new_pos
            # Compute delta using vectorized ops
            vals = lns_A * x_new + lns_B * y_new + lns_C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[i, :].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                pts[i] = new_pos
                current_point_set.discard(old_pos)
                current_point_set.add(new_pos)
                inc[i, :] = now_inc.astype(bool)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Line move
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(new_line[0], new_line[1], new_line[2])
            
            if new_canon is None or new_canon == old_canon:
                continue
            # Check for duplicate
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    k_canon = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                    if k_canon == new_canon:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            A, B, C = new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[:, j].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                lns[j] = new_line
                inc[:, j] = now_inc.astype(bool)
                # Update lns_A, lns_B, lns_C
                lns_A[j] = A
                lns_B[j] = B
                lns_C[j] = C
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
    return best_pts, best_lns, best_total

def _polish_pass(points, lines, deadline):
    """
    A final polish pass that tries to improve each point and line individually.
    """
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
    
    total_inc = int(inc.sum())
    
    # For each point, try all candidate positions and pick the best
    current_point_set = set(map(tuple, pts.tolist()))
    
    # Generate candidate points
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    for p in current_point_set:
        cand_points.add(p)
    cand_points = list(cand_points)
    
    # For each point, find best position
    for i in range(n_pts):
        if time.time() > deadline:
            break
        old_pos = tuple(pts[i])
        best_delta = 0
        best_new_pos = old_pos
        
        for new_pos in cand_points:
            if new_pos == old_pos:
                continue
            # Check if new_pos is already used by another point
            if new_pos in current_point_set:
                if new_pos != old_pos:
                    continue
            
            x_new, y_new = new_pos
            vals = lns[:, 0] * x_new + lns[:, 1] * y_new + lns[:, 2]
            now_inc = (vals == 0).astype(int)
            was_inc = inc[i, :].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            if delta > best_delta:
                best_delta = delta
                best_new_pos = new_pos
        
        if best_delta > 0:
            pts[i] = best_new_pos
            current_point_set.discard(old_pos)
            current_point_set.add(best_new_pos)
            x_new, y_new = best_new_pos
            vals = lns[:, 0] * x_new + lns[:, 1] * y_new + lns[:, 2]
            inc[i, :] = (vals == 0)
            total_inc += best_delta
    
    # For each line, try to improve it
    cand_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 15 and abs(B) <= 15 and abs(C) <= 60:
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    for j in range(n_lines):
        canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        if canon:
            cand_lines.add(canon)
    cand_lines = list(cand_lines)
    
    for j in range(n_lines):
        if time.time() > deadline:
            break
        old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
        best_delta = 0
        best_new_line = None
        
        for new_line in cand_lines:
            new_canon = _normalize_line(new_line[0], new_line[1], new_line[2])
            if new_canon is None or new_canon == old_canon:
                continue
            # Check for duplicate
            is_dup = False
            for k in range(n_lines):
                if k != j:
                    k_canon = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                    if k_canon == new_canon:
                        is_dup = True
                        break
            if is_dup:
                continue
            
            A, B, C = new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[:, j].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            if delta > best_delta:
                best_delta = delta
                best_new_line = new_line
        
        if best_delta > 0 and best_new_line is not None:
            lns[j] = best_new_line
            A, B, C = best_new_line
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            inc[:, j] = (vals == 0)
            total_inc += best_delta
    
    return pts, lns, total_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Direct construction
    pts, lns = _direct_construction_v2()
    
    # Ensure we have exactly 27 lines
    if len(lns) < 27:
        rng = random.Random(1)
        while len(lns) < 27:
            A = rng.randint(-5, 5)
            B = rng.randint(-5, 5)
            if A == 0 and B == 0:
                continue
            C = rng.randint(-20, 20)
            canon = _normalize_line(A, B, C)
            if canon and canon not in set(lns):
                lns.append(canon)
    else:
        lns = lns[:27]
        
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    # Local search
    if time.time() < deadline - 60:
        ls_pts, ls_lns, ls_count = _local_search(pts, lns, deadline - 30)
        pts = ls_pts
        lns = ls_lns
    
    # Polish pass
    if time.time() < deadline - 15:
        pp_pts, pp_lns, pp_count = _polish_pass(pts, lns, deadline - 5)
        pts = pp_pts
        lns = pp_lns
    
    # Final validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback: simple grid
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        lns = lines
    
    return pts, lns
