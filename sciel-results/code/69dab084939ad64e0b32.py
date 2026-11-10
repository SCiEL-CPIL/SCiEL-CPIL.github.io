# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import product, combinations

def normalize_line(A, B, C):
    """Normalize a line so first nonzero coefficient is positive and gcd=1."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    """Count total incidences exactly using int64 arithmetic."""
    total = 0
    for i in range(len(points)):
        x, y = points[i]
        for j in range(len(lines)):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                total += 1
    return total

def fast_incidence_count(points_arr, lines_arr):
    """Vectorized incidence count using numpy."""
    # points_arr: (54, 2), lines_arr: (27, 3)
    # For each point (x,y) and line (A,B,C), check A*x + B*y + C == 0
    x = points_arr[:, 0]  # (54,)
    y = points_arr[:, 1]  # (54,)
    A = lines_arr[:, 0]   # (27,)
    B = lines_arr[:, 1]   # (27,)
    C = lines_arr[:, 2]   # (27,)
    # Compute A*x + B*y + C for all pairs: (54, 27)
    vals = x[:, None] * A[None, :] + y[:, None] * B[None, :] + C[None, :]
    return int(np.sum(vals == 0))

def build_grid_config(n_v, n_h, n_d1, n_d2, x_range, y_range):
    """
    Build a configuration with n_v vertical lines, n_h horizontal lines,
    n_d1 slope-1 lines, n_d2 slope-(-1) lines.
    Points are selected from intersections.
    Returns (points_list, lines_list, score).
    """
    lines = []
    # Vertical lines: x = c
    v_offsets = x_range[:n_v]
    for c in v_offsets:
        lines.append((1, 0, -c))
    # Horizontal lines: y = c
    h_offsets = y_range[:n_h]
    for c in h_offsets:
        lines.append((0, 1, -c))
    # Slope 1 lines: x + y = c  => 1*x + 1*y - c = 0
    d1_sums = sorted(set(x + y for x in x_range for y in y_range))
    # Pick the most common sums
    sum_freq = {}
    for x in x_range:
        for y in y_range:
            s = x + y
            sum_freq[s] = sum_freq.get(s, 0) + 1
    sorted_sums = sorted(sum_freq.keys(), key=lambda s: -sum_freq[s])
    d1_offsets = sorted_sums[:n_d1]
    for c in d1_offsets:
        lines.append((1, 1, -c))
    # Slope -1 lines: x - y = c  => 1*x - 1*y - c = 0
    diff_freq = {}
    for x in x_range:
        for y in y_range:
            d = x - y
            diff_freq[d] = diff_freq.get(d, 0) + 1
    sorted_diffs = sorted(diff_freq.keys(), key=lambda d: -diff_freq[d])
    d2_offsets = sorted_diffs[:n_d2]
    for c in d2_offsets:
        lines.append((1, -1, -c))
    
    # Now select 54 points that maximize incidences
    # Candidate points: all integer points in a reasonable range
    all_candidates = set()
    for x in range(min(x_range) - 3, max(x_range) + 4):
        for y in range(min(y_range) - 3, max(y_range) + 4):
            all_candidates.add((x, y))
    
    # Score each candidate point
    candidate_scores = {}
    for (x, y) in all_candidates:
        cnt = 0
        for (A, B, C) in lines:
            if A * x + B * y + C == 0:
                cnt += 1
        candidate_scores[(x, y)] = cnt
    
    # Greedy selection: pick highest scoring points, but also consider
    # that picking a point might affect... no, points are independent given fixed lines
    # But we want diversity. Let's just pick top 54 by score, breaking ties by position.
    sorted_candidates = sorted(all_candidates, key=lambda p: (-candidate_scores[p], p[0], p[1]))
    points = [list(p) for p in sorted_candidates[:54]]
    
    # Check for duplicates
    unique_pts = set(tuple(p) for p in points)
    if len(unique_pts) != 54:
        # Need to fix
        seen = set()
        fixed_points = []
        for p in sorted_candidates:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                fixed_points.append(list(p))
            if len(fixed_points) == 54:
                break
        points = fixed_points
    
    score = count_incidences(points, lines)
    return points, lines, score

def local_search_points(points, lines, time_limit=30.0, search_radius=3):
    """
    Local search on points only. Lines are fixed.
    Try moving each point to nearby integer positions, plus swap moves.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    current_score = fast_incidence_count(points_arr, lines_arr)
    best_score = current_score
    best_points = points_arr.copy()
    
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    vals = x[:, None] * A[None, :] + y[:, None] * B[None, :] + C[None, :]
    point_line_inc = (vals == 0)
    
    pt_inc_count = point_line_inc.sum(axis=1)
    line_inc_count = point_line_inc.sum(axis=0)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Single-point moves
        for i in range(n_pts):
            if time.time() - start_time > time_limit:
                break
            x0, y0 = int(points_arr[i][0]), int(points_arr[i][1])
            old_inc = int(pt_inc_count[i])
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    # Check duplicate
                    dup = False
                    for k in range(n_pts):
                        if k != i and int(points_arr[k][0]) == nx and int(points_arr[k][1]) == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    new_inc = 0
                    for j in range(n_lines):
                        if A[j] * nx + B[j] * ny + C[j] == 0:
                            new_inc += 1
                    
                    delta = new_inc - old_inc
                    if delta > 0:
                        points_arr[i] = [nx, ny]
                        pt_inc_count[i] = new_inc
                        for j in range(n_lines):
                            was_on = point_line_inc[i, j]
                            is_on = (A[j] * nx + B[j] * ny + C[j] == 0)
                            if was_on and not is_on:
                                line_inc_count[j] -= 1
                            elif is_on and not was_on:
                                line_inc_count[j] += 1
                            point_line_inc[i, j] = is_on
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_points = points_arr.copy()
                        improved = True
                        break
                if improved:
                    break
        
        if time.time() - start_time < time_limit:
            for i in range(n_pts):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_pts):
                    if time.time() - start_time > time_limit:
                        break
                    x0i, y0i = int(points_arr[i][0]), int(points_arr[i][1])
                    x0j, y0j = int(points_arr[j][0]), int(points_arr[j][1])
                    old_inc_i = int(pt_inc_count[i])
                    old_inc_j = int(pt_inc_count[j])
                    
                    for dxi in range(-2, 3):
                        for dyi in range(-2, 3):
                            if dxi == 0 and dyi == 0:
                                continue
                            nxi, nyi = x0i + dxi, y0i + dyi
                            dup_i = False
                            for k in range(n_pts):
                                if k != i and int(points_arr[k][0]) == nxi and int(points_arr[k][1]) == nyi:
                                    dup_i = True
                                    break
                            if dup_i:
                                continue
                            
                            for dxj in range(-2, 3):
                                for dyj in range(-2, 3):
                                    if dxj == 0 and dyj == 0:
                                        continue
                                    nxj, nyj = x0j + dxj, y0j + dyj
                                    dup_j = False
                                    for k in range(n_pts):
                                        if k == i:
                                            if nxi == nxj and nyi == nyj:
                                                dup_j = True
                                                break
                                        elif k != j and int(points_arr[k][0]) == nxj and int(points_arr[k][1]) == nyj:
                                            dup_j = True
                                            break
                                    if dup_j:
                                        continue
                                    
                                    new_inc_i = 0
                                    for jj in range(n_lines):
                                        if A[jj] * nxi + B[jj] * nyi + C[jj] == 0:
                                            new_inc_i += 1
                                    new_inc_j = 0
                                    for jj in range(n_lines):
                                        if A[jj] * nxj + B[jj] * nyj + C[jj] == 0:
                                            new_inc_j += 1
                                    
                                    delta = (new_inc_i - old_inc_i) + (new_inc_j - old_inc_j)
                                    if delta > 0:
                                        points_arr[i] = [nxi, nyi]
                                        points_arr[j] = [nxj, nyj]
                                        pt_inc_count[i] = new_inc_i
                                        pt_inc_count[j] = new_inc_j
                                        for jj in range(n_lines):
                                            was_on_i = point_line_inc[i, jj]
                                            is_on_i = (A[jj] * nxi + B[jj] * nyi + C[jj] == 0)
                                            if was_on_i and not is_on_i:
                                                line_inc_count[jj] -= 1
                                            elif is_on_i and not was_on_i:
                                                line_inc_count[jj] += 1
                                            point_line_inc[i, jj] = is_on_i
                                            
                                            was_on_j = point_line_inc[j, jj]
                                            is_on_j = (A[jj] * nxj + B[jj] * nyj + C[jj] == 0)
                                            if was_on_j and not is_on_j:
                                                line_inc_count[jj] -= 1
                                            elif is_on_j and not was_on_j:
                                                line_inc_count[jj] += 1
                                            point_line_inc[j, jj] = is_on_j
                                        current_score += delta
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_points = points_arr.copy()
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    return best_points.tolist(), best_score

def local_search_lines(points, lines, time_limit=30.0):
    """
    Local search on lines only. Points are fixed.
    Try changing line coefficients slightly.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64).copy()
    
    current_score = fast_incidence_count(points_arr, lines_arr)
    best_score = current_score
    best_lines = lines_arr.copy()
    
    # For each line, try nearby coefficient changes
    for j in range(n_lines):
        if time.time() - start_time > time_limit:
            break
        A0, B0, C0 = int(lines_arr[j][0]), int(lines_arr[j][1]), int(lines_arr[j][2])
        for dA in range(-3, 4):
            for dB in range(-3, 4):
                for dC in range(-8, 9):
                    if dA == 0 and dB == 0 and dC == 0:
                        continue
                    NA, NB, NC = A0 + dA, B0 + dB, C0 + dC
                    if NA == 0 and NB == 0:
                        continue
                    norm_new = normalize_line(NA, NB, NC)
                    is_dup = False
                    for k in range(n_lines):
                        if k != j:
                            norm_k = normalize_line(int(lines_arr[k][0]), int(lines_arr[k][1]), int(lines_arr[k][2]))
                            if norm_new == norm_k:
                                is_dup = True
                                break
                    if is_dup:
                        continue
                    
                    old_line = lines_arr[j].copy()
                    lines_arr[j] = [NA, NB, NC]
                    new_score = fast_incidence_count(points_arr, lines_arr)
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_lines = lines_arr.copy()
                    else:
                        lines_arr[j] = old_line
    
    return best_lines.tolist(), best_score

def joint_local_search(points, lines, time_limit=60.0):
    """
    Alternating local search on points and lines.
    """
    start_time = time.time()
    best_score = fast_incidence_count(np.array(points, dtype=np.int64), np.array(lines, dtype=np.int64))
    best_points = [list(p) for p in points]
    best_lines = [list(l) for l in lines]
    
    while time.time() - start_time < time_limit:
        p_new, s_new = local_search_points(best_points, best_lines, time_limit=15.0, search_radius=3)
        if s_new > best_score:
            best_score = s_new
            best_points = p_new
        
        l_new, s_new = local_search_lines(best_points, best_lines, time_limit=15.0)
        if s_new > best_score:
            best_score = s_new
            best_lines = l_new
        
        if time.time() - start_time > time_limit - 5:
            break
    
    return best_points, best_lines, best_score

def polish_points_for_lines(points, lines, time_limit=30.0):
    """
    Final high-precision polish: try larger moves for under-incidence points.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64)
    
    current_score = fast_incidence_count(points_arr, lines_arr)
    best_score = current_score
    best_points = points_arr.copy()
    
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    vals = x[:, None] * A[None, :] + y[:, None] * B[None, :] + C[None, :]
    point_line_inc = (vals == 0)
    pt_inc_count = point_line_inc.sum(axis=1)
    
    # Build candidate set: all pairwise line intersections (unbounded) + grid around current points
    intersections = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            A1, B1, C1 = int(A[i]), int(B[i]), int(C[i])
            A2, B2, C2 = int(A[j]), int(B[j]), int(C[j])
            det = A1 * B2 - A2 * B1
            if det != 0:
                x_num = B1 * C2 - B2 * C1
                y_num = A2 * C1 - A1 * C2
                if x_num % det == 0 and y_num % det == 0:
                    xi = x_num // det
                    yi = y_num // det
                    # Only keep intersections that are "reasonable" (not astronomically large)
                    if abs(xi) <= 1000 and abs(yi) <= 1000:
                        intersections.add((xi, yi))
    
    # Also add points from existing points (they're valid candidates)
    for i in range(n_pts):
        intersections.add((int(points_arr[i][0]), int(points_arr[i][1])))
    
    for i in range(n_pts):
        if time.time() - start_time > time_limit:
            break
        if int(pt_inc_count[i]) >= 6:
            continue
        
        x0, y0 = int(points_arr[i][0]), int(points_arr[i][1])
        old_inc = int(pt_inc_count[i])
        
        # Use all intersections as candidates, plus a local grid
        candidates = set(intersections)
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                candidates.add((x0 + dx, y0 + dy))
        
        best_new_pos = None
        best_new_inc = old_inc
        
        # Pre-compute a set of occupied positions for fast duplicate check
        occupied = set()
        for k in range(n_pts):
            if k != i:
                occupied.add((int(points_arr[k][0]), int(points_arr[k][1])))
        
        for (cx, cy) in candidates:
            if cx == x0 and cy == y0:
                continue
            if (cx, cy) in occupied:
                continue
            
            new_inc = 0
            for j in range(n_lines):
                if A[j] * cx + B[j] * cy + C[j] == 0:
                    new_inc += 1
            
            if new_inc > best_new_inc:
                best_new_inc = new_inc
                best_new_pos = (cx, cy)
        
        if best_new_pos is not None:
            nx, ny = best_new_pos
            for j in range(n_lines):
                point_line_inc[i, j] = (A[j] * nx + B[j] * ny + C[j] == 0)
            points_arr[i] = [nx, ny]
            pt_inc_count[i] = best_new_inc
            current_score += (best_new_inc - old_inc)
            if current_score > best_score:
                best_score = current_score
                best_points = points_arr.copy()
    
    return best_points.tolist(), best_score

def polish_lines_for_points(points, lines, time_limit=30.0):
    """
    Polish lines given fixed points: for each line, try replacing it with the
    best line (by incidence count) determined by any pair of the 54 points.
    """
    start_time = time.time()
    n_pts = len(points)
    n_lines = len(lines)
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(lines, dtype=np.int64).copy()
    
    current_score = fast_incidence_count(points_arr, lines_arr)
    best_score = current_score
    best_lines = lines_arr.copy()
    
    # Pre-compute all lines determined by pairs of points
    pair_lines = {}  # normalized (A,B,C) -> (A,B,C)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = int(points_arr[i][0]), int(points_arr[i][1])
            x2, y2 = int(points_arr[j][0]), int(points_arr[j][1])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = normalize_line(A, B, C)
            if norm not in pair_lines:
                pair_lines[norm] = norm
    
    all_pair_lines = list(pair_lines.values())
    
    # For each line, find the best replacement from pair_lines
    for j in range(n_lines):
        if time.time() - start_time > time_limit:
            break
        
        old_line = lines_arr[j].copy()
        old_norm = normalize_line(int(old_line[0]), int(old_line[1]), int(old_line[2]))
        
        # Get current normalized set of other lines
        other_norms = set()
        for k in range(n_lines):
            if k != j:
                other_norms.add(normalize_line(int(lines_arr[k][0]), int(lines_arr[k][1]), int(lines_arr[k][2])))
        
        # Try each candidate line
        best_candidate = None
        best_candidate_score = -1
        
        # Also try the current line's score as baseline
        lines_arr[j] = old_line
        baseline_score = fast_incidence_count(points_arr, lines_arr)
        
        for cand in all_pair_lines:
            if time.time() - start_time > time_limit:
                break
            if cand == old_norm:
                continue
            if cand in other_norms:
                continue
            
            lines_arr[j] = cand
            s = fast_incidence_count(points_arr, lines_arr)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = cand
        
        # Decide whether to replace
        if best_candidate is not None and best_candidate_score > baseline_score:
            lines_arr[j] = best_candidate
            current_score = fast_incidence_count(points_arr, lines_arr)
            if current_score > best_score:
                best_score = current_score
                best_lines = lines_arr.copy()
        else:
            lines_arr[j] = old_line
    
    return best_lines.tolist(), best_score

def greedy_point_selection(lines, n_points=54, time_limit=20.0):
    """
    Greedy selection of points to maximize incidences.
    """
    start_time = time.time()
    n_lines = len(lines)
    lines_arr = np.array(lines, dtype=np.int64)
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    candidate_range = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            A1, B1, C1 = int(lines_arr[i][0]), int(lines_arr[i][1]), int(lines_arr[i][2])
            A2, B2, C2 = int(lines_arr[j][0]), int(lines_arr[j][1]), int(lines_arr[j][2])
            det = A1 * B2 - A2 * B1
            if det != 0:
                x_num = B1 * C2 - B2 * C1
                y_num = A2 * C1 - A1 * C2
                if x_num % det == 0 and y_num % det == 0:
                    xi = x_num // det
                    yi = y_num // det
                    candidate_range.add((xi, yi))
    
    for (A_i, B_i, C_i) in lines:
        A_i, B_i, C_i = int(A_i), int(B_i), int(C_i)
        if B_i != 0:
            for x in range(-25, 26):
                y_num = -(A_i * x + C_i)
                if y_num % B_i == 0:
                    y = y_num // B_i
                    if -25 <= y <= 25:
                        candidate_range.add((x, y))
        elif A_i != 0:
            x = -C_i // A_i
            if -25 <= x <= 25:
                for y in range(-25, 26):
                    candidate_range.add((x, y))
    
    candidates = list(candidate_range)
    cand_scores = {}
    for (x, y) in candidates:
        cnt = 0
        for j in range(n_lines):
            if A[j] * x + B[j] * y + C[j] == 0:
                cnt += 1
        cand_scores[(x, y)] = cnt
    
    sorted_cands = sorted(candidates, key=lambda p: (-cand_scores[p], p[0], p[1]))
    points = [list(p) for p in sorted_cands[:n_points]]
    
    seen = set()
    unique_points = []
    for p in sorted_cands:
        if tuple(p) not in seen:
            seen.add(tuple(p))
            unique_points.append(list(p))
        if len(unique_points) == n_points:
            break
    points = unique_points
    
    score = count_incidences(points, lines)
    return points, score

def try_more_grid_configs(deadline):
    """
    Try additional grid configurations with different parameters.
    """
    best_score = 0
    best_points = None
    best_lines = None
    
    grid_sizes = [
        (1, 10), (2, 11), (0, 9), (-1, 8),
        (3, 12), (4, 13),
    ]
    
    splits = [
        (9, 9, 4, 5),
        (9, 9, 5, 4),
        (8, 8, 6, 5),
        (8, 8, 5, 6),
        (9, 8, 5, 5),
        (8, 9, 5, 5),
        (7, 7, 7, 6),
        (7, 7, 6, 7),
        (6, 6, 8, 7),
        (6, 6, 7, 8),
        (10, 10, 3, 4),
        (10, 10, 4, 3),
        (5, 5, 9, 8),
        (5, 5, 8, 9),
    ]
    
    for (start, end) in grid_sizes:
        if time.time() > deadline - 50:
            break
        x_range = list(range(start, end))
        y_range = list(range(start, end))
        
        for (nv, nh, nd1, nd2) in splits:
            if time.time() > deadline - 50:
                break
            if nv + nh + nd1 + nd2 != 27:
                continue
            try:
                points, lines, score = build_grid_config(nv, nh, nd1, nd2, x_range, y_range)
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_lines = lines
            except Exception:
                pass
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_overall_score = 0
    best_overall_points = None
    best_overall_lines = None
    
    x_range = list(range(1, 10))
    y_range = list(range(1, 10))
    
    splits = [
        (9, 9, 4, 5),
        (9, 9, 5, 4),
        (8, 8, 6, 5),
        (8, 8, 5, 6),
        (9, 8, 5, 5),
        (8, 9, 5, 5),
        (7, 7, 7, 6),
        (7, 7, 6, 7),
        (6, 6, 8, 7),
        (6, 6, 7, 8),
    ]
    
    for (nv, nh, nd1, nd2) in splits:
        if time.time() > deadline - 120:
            break
        points, lines, score = build_grid_config(nv, nh, nd1, nd2, x_range, y_range)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_lines = lines
    
    if time.time() < deadline - 100:
        extra_pts, extra_lines, extra_score = try_more_grid_configs(deadline)
        if extra_score > best_overall_score:
            best_overall_score = extra_score
            best_overall_points = extra_pts
            best_overall_lines = extra_lines
    
    if best_overall_lines and time.time() < deadline - 80:
        greedy_points, greedy_score = greedy_point_selection(best_overall_lines, n_points=54, time_limit=15.0)
        if greedy_score > best_overall_score:
            best_overall_score = greedy_score
            best_overall_points = greedy_points
    
    if best_overall_points and best_overall_lines and time.time() < deadline - 70:
        ls_points, ls_lines, ls_score = joint_local_search(
            best_overall_points, best_overall_lines, time_limit=35.0
        )
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_points = ls_points
            best_overall_lines = ls_lines
    
    if best_overall_points and best_overall_lines and time.time() < deadline - 50:
        polish_pts, polish_score = polish_points_for_lines(
            best_overall_points, best_overall_lines, time_limit=15.0
        )
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_points = polish_pts
    
    if best_overall_points and best_overall_lines and time.time() < deadline - 30:
        # New: polish lines given fixed points
        polish_lns, polish_lns_score = polish_lines_for_points(
            best_overall_points, best_overall_lines, time_limit=15.0
        )
        if polish_lns_score > best_overall_score:
            best_overall_score = polish_lns_score
            best_overall_lines = polish_lns
    
    if best_overall_points and best_overall_lines and time.time() < deadline - 15:
        # Final round: polish points again after line changes
        polish_pts2, polish_score2 = polish_points_for_lines(
            best_overall_points, best_overall_lines, time_limit=8.0
        )
        if polish_score2 > best_overall_score:
            best_overall_score = polish_score2
            best_overall_points = polish_pts2
        
        # And lines again
        if time.time() < deadline - 5:
            polish_lns2, polish_lns_score2 = polish_lines_for_points(
                best_overall_points, best_overall_lines, time_limit=5.0
            )
            if polish_lns_score2 > best_overall_score:
                best_overall_score = polish_lns_score2
                best_overall_lines = polish_lns2
    
    points_arr = np.array(best_overall_points, dtype=np.int64)
    lines_arr = np.array(best_overall_lines, dtype=np.int64)
    
    assert points_arr.shape == (54, 2)
    assert lines_arr.shape == (27, 3)
    
    unique_pts = set(map(tuple, points_arr.tolist()))
    if len(unique_pts) != 54:
        points, lines, score = build_grid_config(9, 9, 4, 5, x_range, y_range)
        points_arr = np.array(points, dtype=np.int64)
        lines_arr = np.array(lines, dtype=np.int64)
    
    norm_lines = set()
    for l in lines_arr:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        norm_lines.add(normalize_line(A, B, C))
    if len(norm_lines) != 27:
        points, lines, score = build_grid_config(9, 9, 4, 5, x_range, y_range)
        points_arr = np.array(points, dtype=np.int64)
        lines_arr = np.array(lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
