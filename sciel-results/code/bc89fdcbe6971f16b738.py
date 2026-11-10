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

def _build_candidate_lines(max_coeff=8, max_const=25):
    lines = set()
    for A in range(-max_coeff, max_coeff + 1):
        for B in range(-max_coeff, max_coeff + 1):
            if A == 0 and B == 0:
                continue
            for C in range(-max_const, max_const + 1):
                canon = _normalize_line(A, B, C)
                if canon:
                    lines.add(canon)
    return list(lines)

def _compute_pairwise_intersections(lines):
    point_lines = defaultdict(set)
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pt = _line_intersection(lines[i], lines[j])
            if pt is not None:
                point_lines[pt].add(i)
                point_lines[pt].add(j)
    return point_lines

def _generate_lattice_lines_through_points(points, max_slope=4):
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
            canon = _normalize_line(A, B, C)
            if canon:
                if abs(canon[0]) <= 10 and abs(canon[1]) <= 10 and abs(canon[2]) <= 50:
                    lines.add(canon)
    return list(lines)

def _evaluate_configuration(points, lines):
    pts = np.array(points)
    lns = np.array(lines)
    inc_count = 0
    point_lines = defaultdict(set)
    for i, pt in enumerate(pts):
        x, y = pt
        for j, line in enumerate(lns):
            A, B, C = line
            if A * x + B * y + C == 0:
                inc_count += 1
                point_lines[pt].add(j)
    return inc_count, point_lines

def _greedy_line_selection_v2(pool, n_lines=27, points=None):
    """
    Greedy line selection that considers both:
    1. Number of points on each line (if points provided)
    2. Pairwise intersection density with already-selected lines
    
    This creates a more coherent line arrangement where lines intersect at common points.
    """
    if points is None:
        # Fallback: just pick lines with small coefficients
        selected = []
        for line in sorted(pool, key=lambda l: abs(l[0]) + abs(l[1]) + abs(l[2])):
            if len(selected) >= n_lines:
                break
            selected.append(line)
        return selected
    
    pts = np.array(points)
    n_pts = len(pts)
    
    # Precompute point-line incidences for all candidate lines
    line_point_counts = {}
    line_point_sets = {}
    for line in pool:
        A, B, C = line
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        mask = vals == 0
        count = int(mask.sum())
        line_point_counts[line] = count
        line_point_sets[line] = set(np.where(mask)[0].tolist())
    
    # Start with the line that has the most points
    best_line = max(pool, key=lambda l: line_point_counts[l])
    selected = [best_line]
    remaining = [l for l in pool if l != best_line]
    
    # Greedily add lines that maximize:
    # - Point count (primary)
    # - Number of intersection points with already-selected lines (secondary)
    while len(selected) < n_lines and remaining:
        best_score = -1
        best_idx = -1
        for idx, line in enumerate(remaining):
            # Primary: point count
            score = line_point_counts[line]
            
            # Secondary: count intersections with selected lines
            inter_count = 0
            for sel_line in selected:
                pt = _line_intersection(line, sel_line)
                if pt is not None:
                    inter_count += 1
                    # Bonus if this intersection point already has multiple lines
                    # (we can approximate this later)
            
            # Combined score: weight point count heavily, but reward intersections
            # This encourages lines that pass through many points AND intersect others
            combined = score * 10 + inter_count
            
            if combined > best_score:
                best_score = combined
                best_idx = idx
        
        if best_idx == -1:
            break
        
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)
    
    return selected

def _direct_construction_v3():
    """
    Improved direct construction:
    1. Define 54 points on a 6x9 grid
    2. Generate candidate lines
    3. Use greedy selection that considers intersection density
    4. This should create a more coherent arrangement
    """
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    candidate_lines = _generate_lattice_lines_through_points(points, max_slope=3)
    
    # Use improved greedy selection
    selected_lines = _greedy_line_selection_v2(candidate_lines, n_lines=27, points=points)
    
    # Ensure we have exactly 27 lines
    if len(selected_lines) < 27:
        # Pad with additional lines from the pool
        existing = set(selected_lines)
        for line in candidate_lines:
            if line not in existing and len(selected_lines) < 27:
                selected_lines.append(line)
                existing.add(line)
    
    return points, selected_lines[:27]

def _local_search_fast(points, lines, deadline, max_iters=None):
    """
    Fast local search with vectorized delta calculations.
    """
    rng = random.Random(42)
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix (vectorized)
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for j in range(n_lines):
        A, B, C = lns[j]
        inc[:, j] = (A * pts[:, 0] + B * pts[:, 1] + C == 0)
    
    total_inc = int(inc.sum())
    best_total = total_inc
    best_pts = pts.copy()
    best_lns = lns.copy()
    
    # Build candidate points: intersections of current lines + neighbors
    cand_points = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            pt = _line_intersection(tuple(lns[i]), tuple(lns[j]))
            if pt:
                cand_points.add(pt)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        cand_points.add((pt[0] + dx, pt[1] + dy))
    
    # Also add current points as candidates
    for i in range(n_pts):
        cand_points.add(tuple(pts[i]))
    
    # Build candidate lines
    cand_lines = set()
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0: continue
            A = dy // g
            B = -dx // g
            C = -(A * x1 + B * y1)
            if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                canon = _normalize_line(A, B, C)
                if canon:
                    cand_lines.add(canon)
    
    for _ in range(200):
        A = rng.randint(-5, 5)
        B = rng.randint(-5, 5)
        if A == 0 and B == 0: continue
        C = rng.randint(-20, 20)
        canon = _normalize_line(A, B, C)
        if canon:
            cand_lines.add(canon)
            
    cand_points = list(cand_points)
    cand_lines = list(cand_lines)
    
    if not cand_points or not cand_lines:
        return best_pts, best_lns, best_total

    T_start = 5.0
    T_end = 0.01
    if max_iters is None:
        max_iters = 80000  # Increased due to faster per-iteration cost
    iter_count = 0
    no_improve_count = 0
    no_improve_limit = 5000
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        T = T_start * (T_end / T_start) ** (min(iter_count, max_iters) / max_iters)
        
        if rng.random() < 0.5:
            # Point move (vectorized delta)
            i = rng.randint(0, n_pts - 1)
            old_pos = tuple(pts[i])
            new_pos = cand_points[rng.randint(0, len(cand_points) - 1)]
            
            if new_pos == old_pos:
                continue
            if any(tuple(pts[k]) == new_pos for k in range(n_pts) if k != i):
                continue
            
            # Vectorized delta calculation
            new_vals = lns[:, 0] * new_pos[0] + lns[:, 1] * new_pos[1] + lns[:, 2]
            new_inc = (new_vals == 0).astype(int)
            old_inc = inc[i, :].astype(int)
            delta = int(new_inc.sum() - old_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                pts[i] = new_pos
                inc[i, :] = (new_vals == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Line move (vectorized delta)
            j = rng.randint(0, n_lines - 1)
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            new_line = cand_lines[rng.randint(0, len(cand_lines) - 1)]
            new_canon = _normalize_line(int(new_line[0]), int(new_line[1]), int(new_line[2]))
            
            if new_canon is None or new_canon == old_canon:
                continue
            # Check for duplicate lines (fast path)
            dup_found = False
            for k in range(n_lines):
                if k != j:
                    ck = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                    if ck == new_canon:
                        dup_found = True
                        break
            if dup_found:
                continue
            
            # Vectorized delta calculation
            A, B, C = new_line
            new_vals = A * pts[:, 0] + B * pts[:, 1] + C
            new_inc = (new_vals == 0).astype(int)
            old_inc = inc[:, j].astype(int)
            delta = int(new_inc.sum() - old_inc.sum())
            
            new_total = total_inc + delta
            if new_total >= total_inc or rng.random() < np.exp(delta / max(T, 1e-10)):
                total_inc = new_total
                lns[j] = new_line
                inc[:, j] = (new_vals == 0)
                
                if total_inc > best_total:
                    best_total = total_inc
                    best_pts = pts.copy()
                    best_lns = lns.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        # Early stopping if no improvement for a while
        if no_improve_count > no_improve_limit and iter_count > max_iters // 4:
            break
    
    return best_pts, best_lns, best_total

def _polish_pass(points, lines, deadline):
    """
    Final high-precision polish: try all single-point moves and all single-line moves
    to see if any improve the score. Repeat until no improvement or deadline.
    """
    pts = np.array(points, dtype=np.int64)
    lns = np.array(lines, dtype=np.int64)
    n_pts = 54
    n_lines = 27
    
    # Build incidence matrix
    inc = np.zeros((n_pts, n_lines), dtype=bool)
    for j in range(n_lines):
        A, B, C = lns[j]
        inc[:, j] = (A * pts[:, 0] + B * pts[:, 1] + C == 0)
    
    total_inc = int(inc.sum())
    improved = True
    max_rounds = 20
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try moving each point to nearby positions
        for i in range(n_pts):
            if time.time() > deadline:
                break
            x, y = pts[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = (x + dx, y + dy)
                    # Check for duplicates
                    if any(tuple(pts[k]) == new_pos for k in range(n_pts) if k != i):
                        continue
                    
                    # Calculate delta
                    new_vals = lns[:, 0] * new_pos[0] + lns[:, 1] * new_pos[1] + lns[:, 2]
                    new_inc = (new_vals == 0).astype(int)
                    old_inc = inc[i, :].astype(int)
                    delta = int(new_inc.sum() - old_inc.sum())
                    
                    if delta > 0:
                        pts[i] = new_pos
                        inc[i, :] = (new_vals == 0)
                        total_inc += delta
                        improved = True
                        break
                if improved and dx != 0:
                    break
            if improved:
                continue
        
        # Try replacing each line
        for j in range(n_lines):
            if time.time() > deadline:
                break
            if not improved:
                break
            
            old_canon = _normalize_line(int(lns[j, 0]), int(lns[j, 1]), int(lns[j, 2]))
            
            # Generate candidate lines through pairs of points
            candidates = []
            for i1 in range(n_pts):
                for i2 in range(i1 + 1, n_pts):
                    x1, y1 = pts[i1]
                    x2, y2 = pts[i2]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0: continue
                    A = dy // g
                    B = -dx // g
                    C = -(A * x1 + B * y1)
                    if abs(A) <= 10 and abs(B) <= 10 and abs(C) <= 50:
                        canon = _normalize_line(A, B, C)
                        if canon and canon != old_canon:
                            candidates.append(canon)
            
            # Also try some random lines
            rng = random.Random(j + round_num)
            for _ in range(50):
                A = rng.randint(-5, 5)
                B = rng.randint(-5, 5)
                if A == 0 and B == 0: continue
                C = rng.randint(-20, 20)
                canon = _normalize_line(A, B, C)
                if canon and canon != old_canon:
                    candidates.append(canon)
            
            best_delta = 0
            best_line = None
            for cand in candidates:
                # Check for duplicates
                dup = False
                for k in range(n_lines):
                    if k != j:
                        ck = _normalize_line(int(lns[k, 0]), int(lns[k, 1]), int(lns[k, 2]))
                        if ck == cand:
                            dup = True
                            break
                if dup:
                    continue
                
                A, B, C = cand
                new_vals = A * pts[:, 0] + B * pts[:, 1] + C
                new_inc = (new_vals == 0).astype(int)
                old_inc = inc[:, j].astype(int)
                delta = int(new_inc.sum() - old_inc.sum())
                
                if delta > best_delta:
                    best_delta = delta
                    best_line = cand
            
            if best_line is not None and best_delta > 0:
                lns[j] = best_line
                A, B, C = best_line
                inc[:, j] = (A * pts[:, 0] + B * pts[:, 1] + C == 0)
                total_inc += best_delta
                improved = True
    
    return pts, lns, total_inc

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 570  # Safety margin of 30s before 600s limit
    
    # Direct construction with improved greedy selection
    pts, lns = _direct_construction_v3()
    
    # Ensure we have exactly 27 lines
    if len(lns) < 27:
        rng = random.Random(1)
        existing = set(lns)
        while len(lns) < 27:
            A = rng.randint(-5, 5)
            B = rng.randint(-5, 5)
            if A == 0 and B == 0: continue
            C = rng.randint(-20, 20)
            canon = _normalize_line(A, B, C)
            if canon and canon not in existing:
                lns.append(canon)
                existing.add(canon)
    else:
        lns = lns[:27]
        
    pts = np.array(pts, dtype=np.int64)
    lns = np.array(lns, dtype=np.int64)
    
    # Fast local search
    if time.time() < deadline - 60:
        ls_pts, ls_lns, ls_count = _local_search_fast(pts, lns, deadline - 30)
        pts = ls_pts
        lns = ls_lns
    
    # Final polish pass
    if time.time() < deadline - 10:
        pol_pts, pol_lns, pol_count = _polish_pass(pts, lns, deadline - 5)
        pts = pol_pts
        lns = pol_lns
    
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
