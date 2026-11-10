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
    # Vectorized incidence calculation
    # lns: (27, 3), pts: (54, 2)
    # We want to check A*x + B*y + C == 0 for all pairs
    # Expand dims: lns[:, 0, None] * pts[:, 0, None] ... 
    # This creates a (27, 1, 54) array? No.
    # Let's do: lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]
    # lns[:, 0:1] is (27, 1)
    # pts[:, 0:1].T is (1, 54)
    # Product is (27, 54)
    vals = lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]
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
    if abs(x) > 100 or abs(y) > 100:
        return None
    return (x, y)

def _generate_lattice_lines_through_points(points, max_coeff=12, max_const=60):
    """
    Generate all lines passing through at least 2 points.
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

def _direct_construction_grid(r, c):
    """
    Construct a grid of size r x c.
    """
    points = []
    for x in range(r):
        for y in range(c):
            points.append((x, y))
    
    # If we have fewer than 54 points, pad with random nearby points?
    # If more, trim?
    # For now, assume r*c == 54 or we handle it in the caller.
    
    candidate_lines = _generate_lattice_lines_through_points(points, max_coeff=10, max_const=50)
    
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
    
    if len(selected_lines) < 27:
        # Pad with random lines
        rng = random.Random(123)
        while len(selected_lines) < 27:
            A = rng.randint(-5, 5)
            B = rng.randint(-5, 5)
            if A == 0 and B == 0:
                continue
            C = rng.randint(-20, 20)
            canon = _normalize_line(A, B, C)
            if canon and canon not in seen:
                selected_lines.append(canon)
                seen.add(canon)
            
    return points, selected_lines

def _local_search(points, lines, deadline):
    """
    Robust local search with simulated annealing.
    Moves:
    1. Move a point to a new position.
    2. Change a line to a new line.
    3. Swap two points.
    4. Swap two lines.
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Precompute incidence matrix
    # inc[i, j] is True if point i is on line j
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for i in range(n_pts):
        for j in range(n_lines):
            A, B, C = lns[j]
            inc[i, j] = (A * pts[i, 0] + B * pts[i, 1] + C == 0)
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Candidate points: Intersections of lines + neighbors
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                # Add neighbors to allow "breathing"
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    # Add current points
    for p in pts:
        cand_points.add(tuple(p))
    # Add some random points in a reasonable box
    for _ in range(500):
        cand_points.add((rng.randint(-10, 20), rng.randint(-10, 20)))
    cand_points = list(cand_points)
    
    # Candidate lines: Lines through pairs of points + random
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
    # Add random lines
    for _ in range(300):
        A = rng.randint(-8, 8)
        B = rng.randint(-8, 8)
        if A == 0 and B == 0:
            continue
        C = rng.randint(-30, 30)
        canon = _normalize_line(A, B, C)
        if canon:
            cand_lines.add(canon)
    cand_lines = list(cand_lines)
    
    # Precompute line coeffs for speed
    lns_A = lns[:, 0].astype(np.int64)
    lns_B = lns[:, 1].astype(np.int64)
    lns_C = lns[:, 2].astype(np.int64)
    
    T_start = 10.0
    T_end = 0.01
    n_iters = 200000
    iter_count = 0
    no_improve = 0
    max_no_improve = 10000
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, n_iters) / n_iters)
        
        move_type = rng.randint(0, 3)
        
        if move_type == 0:
            # Point move
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            # Check if new_pos is already used
            # This is O(N), which is slow. We can use a set of current points.
            # Let's maintain a set of current points.
            # For simplicity in this snippet, I'll do a linear check but it's bad.
            # Better: maintain a dict of point->index? No, points can be duplicated in cand_points but not in pts.
            # Let's just check if any other point is at new_pos.
            is_dup = False
            for k in range(n_pts):
                if k != i and tuple(pts[k]) == new_pos:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            x_new, y_new = new_pos
            vals = lns_A * x_new + lns_B * y_new + lns_C
            now_inc = (vals == 0).astype(int)
            was_inc = inc[i, :].astype(int)
            delta = int(now_inc.sum() - was_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                pts[i] = new_pos
                inc[i, :] = now_inc.astype(bool)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        elif move_type == 1:
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
        elif move_type == 2:
            # Swap two points
            i1, i2 = rng.sample(range(n_pts), 2)
            p1 = tuple(pts[i1])
            p2 = tuple(pts[i2])
            if p1 == p2:
                continue
            
            # Calculate delta for swapping
            # Point i1 moves to p2, Point i2 moves to p1
            x1, y1 = p1
            x2, y2 = p2
            
            # Incidence for i1 at p2
            vals_i1_new = lns_A * x2 + lns_B * y2 + lns_C
            inc_i1_new = (vals_i1_new == 0).astype(int)
            # Incidence for i2 at p1
            vals_i2_new = lns_A * x1 + lns_B * y1 + lns_C
            inc_i2_new = (vals_i2_new == 0).astype(int)
            
            was_i1 = inc[i1, :].astype(int)
            was_i2 = inc[i2, :].astype(int)
            
            delta = int(inc_i1_new.sum() + inc_i2_new.sum() - was_i1.sum() - was_i2.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                pts[i1] = p2
                pts[i2] = p1
                inc[i1, :] = inc_i1_new.astype(bool)
                inc[i2, :] = inc_i2_new.astype(bool)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Swap two lines
            j1, j2 = rng.sample(range(n_lines), 2)
            l1 = tuple(lns[j1])
            l2 = tuple(lns[j2])
            if l1 == l2:
                continue
            
            # Calculate delta
            # Line j1 becomes l2, Line j2 becomes l1
            A1, B1, C1 = l1
            A2, B2, C2 = l2
            
            # Incidence of l2 on all points (for j1)
            vals_l2 = A2 * pts[:, 0] + B2 * pts[:, 1] + C2
            inc_l2 = (vals_l2 == 0).astype(int)
            # Incidence of l1 on all points (for j2)
            vals_l1 = A1 * pts[:, 0] + B1 * pts[:, 1] + C1
            inc_l1 = (vals_l1 == 0).astype(int)
            
            was_j1 = inc[:, j1].astype(int)
            was_j2 = inc[:, j2].astype(int)
            
            delta = int(inc_l2.sum() + inc_l1.sum() - was_j1.sum() - was_j2.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or (delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10))):
                total_inc = new_total
                lns[j1] = l2
                lns[j2] = l1
                inc[:, j1] = inc_l2.astype(bool)
                inc[:, j2] = inc_l1.astype(bool)
                lns_A[j1] = A2
                lns_B[j1] = B2
                lns_C[j1] = C2
                lns_A[j2] = A1
                lns_B[j2] = B1
                lns_C[j2] = C1
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
    return best_pts, best_lns, best_total

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 600
    
    # Try 6x9 grid
    pts, lns = _direct_construction_grid(6, 9)
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    # Run local search
    if time.time() < deadline - 30:
        ls_pts, ls_lns, ls_count = _local_search(pts, lns, deadline - 10)
        pts = ls_pts
        lns = ls_lns
    
    # Final validation
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    if not _is_valid(pts, lns):
        # Fallback
        xs = np.arange(6)
        ys = np.arange(9)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines = np.array([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)] + [(1, 1, -k) for k in range(12)], dtype=np.int64)
        lns = lines
    
    return pts, lns
