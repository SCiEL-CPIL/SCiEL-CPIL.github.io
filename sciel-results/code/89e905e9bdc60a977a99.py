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
    for x in range(min(x_range) - 2, max(x_range) + 3):
        for y in range(min(y_range) - 2, max(y_range) + 3):
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
    
    # Precompute for each point, which lines it's on
    # And for each line, which points are on it
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    vals = x[:, None] * A[None, :] + y[:, None] * B[None, :] + C[None, :]
    point_line_inc = (vals == 0)
    
    # Incidence count per point
    pt_inc_count = point_line_inc.sum(axis=1)
    # Incidence count per line
    line_inc_count = point_line_inc.sum(axis=0)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Single-point moves
        for i in range(n_pts):
            if time.time() - start_time > time_limit:
                break
            x0, y0 = points_arr[i]
            old_inc = pt_inc_count[i]
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    # Check duplicate
                    dup = False
                    for k in range(n_pts):
                        if k != i and points_arr[k][0] == nx and points_arr[k][1] == ny:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    # Compute new incidence for this point
                    new_inc = 0
                    for j in range(n_lines):
                        if A[j] * nx + B[j] * ny + C[j] == 0:
                            new_inc += 1
                    
                    delta = new_inc - old_inc
                    if delta > 0:
                        # Accept
                        points_arr[i] = [nx, ny]
                        pt_inc_count[i] = new_inc
                        # Update line incidence counts
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
        
        # Swap moves: try moving two points simultaneously
        if time.time() - start_time < time_limit:
            for i in range(n_pts):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_pts):
                    if time.time() - start_time > time_limit:
                        break
                    x0i, y0i = points_arr[i]
                    x0j, y0j = points_arr[j]
                    old_inc_i = pt_inc_count[i]
                    old_inc_j = pt_inc_count[j]
                    
                    # Try small moves for both points
                    for dxi in range(-2, 3):
                        for dyi in range(-2, 3):
                            if dxi == 0 and dyi == 0:
                                continue
                            nxi, nyi = x0i + dxi, y0i + dyi
                            # Check if target for i is occupied by j or others
                            dup_i = False
                            for k in range(n_pts):
                                if k != i and points_arr[k][0] == nxi and points_arr[k][1] == nyi:
                                    dup_i = True
                                    break
                            if dup_i:
                                continue
                            
                            for dxj in range(-2, 3):
                                for dyj in range(-2, 3):
                                    if dxj == 0 and dyj == 0:
                                        continue
                                    nxj, nyj = x0j + dxj, y0j + dyj
                                    # Check if target for j is occupied (including i's new position)
                                    dup_j = False
                                    for k in range(n_pts):
                                        if k == i:
                                            if nxi == nxj and nyi == nyj:
                                                dup_j = True
                                                break
                                        elif k != j and points_arr[k][0] == nxj and points_arr[k][1] == nyj:
                                            dup_j = True
                                            break
                                    if dup_j:
                                        continue
                                    
                                    # Compute new incidences
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
                                        # Accept swap
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
        A0, B0, C0 = lines_arr[j]
        # Try small changes to A, B, C
        for dA in range(-2, 3):
            for dB in range(-2, 3):
                for dC in range(-5, 6):
                    if dA == 0 and dB == 0 and dC == 0:
                        continue
                    NA, NB, NC = A0 + dA, B0 + dB, C0 + dC
                    if NA == 0 and NB == 0:
                        continue
                    # Check if this line is already in the set
                    norm_new = normalize_line(NA, NB, NC)
                    is_dup = False
                    for k in range(n_lines):
                        if k != j:
                            norm_k = normalize_line(lines_arr[k][0], lines_arr[k][1], lines_arr[k][2])
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
        # Optimize points
        p_new, s_new = local_search_points(best_points, best_lines, time_limit=15.0, search_radius=2)
        if s_new > best_score:
            best_score = s_new
            best_points = p_new
        
        # Optimize lines
        l_new, s_new = local_search_lines(best_points, best_lines, time_limit=15.0)
        if s_new > best_score:
            best_score = s_new
            best_lines = l_new
        
        # Check if we can continue
        if time.time() - start_time > time_limit - 5:
            break
        
        # If no improvement in last round, try a different radius
        # For now, just continue
    
    return best_points, best_lines, best_score

def polish_points_for_lines(points, lines, time_limit=30.0):
    """
    Final high-precision polish: for points with low incidence, try larger moves
    and also try to place points at line intersections.
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
    
    # Compute current incidence per point
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    vals = x[:, None] * A[None, :] + y[:, None] * B[None, :] + C[None, :]
    point_line_inc = (vals == 0)
    pt_inc_count = point_line_inc.sum(axis=1)
    
    # Find all line intersections (integer points)
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
                    # Only consider reasonable range
                    if -30 <= xi <= 30 and -30 <= yi <= 30:
                        intersections.add((xi, yi))
    
    # For each point with low incidence, try to move it to a high-incidence location
    for i in range(n_pts):
        if time.time() - start_time > time_limit:
            break
        if pt_inc_count[i] >= 4:  # Only bother with low-incidence points
            continue
        
        x0, y0 = int(points_arr[i][0]), int(points_arr[i][1])
        old_inc = int(pt_inc_count[i])
        
        # Try all intersection points and nearby grid points
        candidates = set(intersections)
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                candidates.add((x0 + dx, y0 + dy))
        
        best_new_pos = None
        best_new_inc = old_inc
        
        for (cx, cy) in candidates:
            if cx == x0 and cy == y0:
                continue
            # Check duplicate
            dup = False
            for k in range(n_pts):
                if k != i and points_arr[k][0] == cx and points_arr[k][1] == cy:
                    dup = True
                    break
            if dup:
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
            # Update
            for j in range(n_lines):
                was_on = point_line_inc[i, j]
                is_on = (A[j] * nx + B[j] * ny + C[j] == 0)
                if was_on and not is_on:
                    pass
                elif is_on and not was_on:
                    pass
                point_line_inc[i, j] = is_on
            points_arr[i] = [nx, ny]
            pt_inc_count[i] = best_new_inc
            current_score += (best_new_inc - old_inc)
            if current_score > best_score:
                best_score = current_score
                best_points = points_arr.copy()
    
    return best_points.tolist(), best_score

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
    
    # Determine candidate range
    # Use the range covered by line intersections
    candidate_range = set()
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            A1, B1, C1 = lines_arr[i]
            A2, B2, C2 = lines_arr[j]
            det = A1 * B2 - A2 * B1
            if det != 0:
                x = (B1 * C2 - B2 * C1) / det
                y = (A2 * C1 - A1 * C2) / det
                if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                    xi, yi = int(round(x)), int(round(y))
                    candidate_range.add((xi, yi))
    
    # Also add points from the bounding box of lines
    for (A_i, B_i, C_i) in lines:
        if B_i != 0:
            # y = -(A_i*x + C_i)/B_i
            for x in range(-20, 21):
                y_num = -(A_i * x + C_i)
                if y_num % B_i == 0:
                    y = y_num // B_i
                    if -20 <= y <= 20:
                        candidate_range.add((x, y))
        elif A_i != 0:
            x = -C_i // A_i
            if -20 <= x <= 20:
                for y in range(-20, 21):
                    candidate_range.add((x, y))
    
    # Score all candidates
    candidates = list(candidate_range)
    cand_scores = {}
    for (x, y) in candidates:
        cnt = 0
        for j in range(n_lines):
            if A[j] * x + B[j] * y + C[j] == 0:
                cnt += 1
        cand_scores[(x, y)] = cnt
    
    # Greedy: pick highest, then update... but since points are independent,
    # just pick top 54
    sorted_cands = sorted(candidates, key=lambda p: (-cand_scores[p], p[0], p[1]))
    points = [list(p) for p in sorted_cands[:n_points]]
    
    # Ensure no duplicates
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

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a multi-phase approach:
    1. Build initial configuration from 4-direction grid
    2. Greedy point selection
    3. Joint local search
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Phase 1: Build initial 4-direction configuration
    x_range = list(range(1, 10))
    y_range = list(range(1, 10))
    
    # Try different splits of 27 lines among 4 directions
    best_overall_score = 0
    best_overall_points = None
    best_overall_lines = None
    
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
        if time.time() > deadline - 100:
            break
        points, lines, score = build_grid_config(nv, nh, nd1, nd2, x_range, y_range)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_lines = lines
    
    # Phase 2: Greedy point selection from best lines
    if best_overall_lines and time.time() < deadline - 80:
        greedy_points, greedy_score = greedy_point_selection(best_overall_lines, n_points=54, time_limit=15.0)
        if greedy_score > best_overall_score:
            best_overall_score = greedy_score
            best_overall_points = greedy_points
    
    # Phase 3: Joint local search
    if best_overall_points and best_overall_lines and time.time() < deadline - 60:
        ls_points, ls_lines, ls_score = joint_local_search(
            best_overall_points, best_overall_lines, time_limit=45.0
        )
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_points = ls_points
            best_overall_lines = ls_lines
    
    # Phase 4: Polish pass with larger moves for under-incidence points
    if best_overall_points and best_overall_lines and time.time() < deadline - 30:
        polish_pts, polish_score = polish_points_for_lines(
            best_overall_points, best_overall_lines, time_limit=20.0
        )
        if polish_score > best_overall_score:
            best_overall_score = polish_score
            best_overall_points = polish_pts
    
    # Phase 5: Another round of greedy + local search + polish
    if best_overall_points and best_overall_lines and time.time() < deadline - 20:
        greedy_points2, greedy_score2 = greedy_point_selection(best_overall_lines, n_points=54, time_limit=10.0)
        if greedy_score2 > best_overall_score:
            best_overall_score = greedy_score2
            best_overall_points = greedy_points2
        
        ls_points2, ls_lines2, ls_score2 = joint_local_search(
            best_overall_points, best_overall_lines, time_limit=20.0
        )
        if ls_score2 > best_overall_score:
            best_overall_score = ls_score2
            best_overall_points = ls_points2
            best_overall_lines = ls_lines2
        
        # Final polish
        if time.time() < deadline - 10:
            polish_pts2, polish_score2 = polish_points_for_lines(
                best_overall_points, best_overall_lines, time_limit=10.0
            )
            if polish_score2 > best_overall_score:
                best_overall_score = polish_score2
                best_overall_points = polish_pts2
    
    # Final validation and conversion
    points_arr = np.array(best_overall_points, dtype=np.int64)
    lines_arr = np.array(best_overall_lines, dtype=np.int64)
    
    # Validate
    assert points_arr.shape == (54, 2)
    assert lines_arr.shape == (27, 3)
    
    # Check for duplicate points
    unique_pts = set(map(tuple, points_arr.tolist()))
    if len(unique_pts) != 54:
        # Fallback
        points, lines, score = build_grid_config(9, 9, 4, 5, x_range, y_range)
        points_arr = np.array(points, dtype=np.int64)
        lines_arr = np.array(lines, dtype=np.int64)
    
    # Check for duplicate lines
    norm_lines = set()
    for l in lines_arr:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        norm_lines.add(normalize_line(A, B, C))
    if len(norm_lines) != 27:
        # Remove duplicates and replace
        seen = set()
        unique_lines = []
        for l in lines_arr:
            A, B, C = int(l[0]), int(l[1]), int(l[2])
            norm = normalize_line(A, B, C)
            if norm not in seen:
                seen.add(norm)
                unique_lines.append([A, B, C])
        # Need to add more lines to reach 27
        # For now, fallback
        points, lines, score = build_grid_config(9, 9, 4, 5, x_range, y_range)
        points_arr = np.array(points, dtype=np.int64)
        lines_arr = np.array(lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
