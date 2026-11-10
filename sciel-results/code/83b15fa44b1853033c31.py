# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
import time


def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x1 == x2:
        return (1, 0, -x1)
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _count_incidences(points, lines):
    total = 0
    for (A, B, C) in lines:
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.sum(vals == 0))
    return total


def _greedy_select_lines(points, m_lines):
    n = len(points)
    candidate_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line != (0, 0, 0):
                candidate_lines.add(line)
    
    # Add small-coeff lines
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                candidate_lines.add(_normalize_line(A, B, C))
    
    candidates = list(candidate_lines)
    # Filter: keep lines with >= 2 incidences
    filtered = []
    for line in candidates:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        if int(np.sum(vals == 0)) >= 2:
            filtered.append(line)
    candidates = filtered
    
    selected = []
    selected_set = set()
    
    for _ in range(m_lines):
        best_line = None
        best_gain = -1
        for line in candidates:
            if line in selected_set:
                continue
            A, B, C = line
            vals = A * points[:, 0] + B * points[:, 1] + C
            cnt = int(np.sum(vals == 0))
            if cnt > best_gain:
                best_gain = cnt
                best_line = line
        if best_line is None or best_gain <= 0:
            break
        selected.append(best_line)
        selected_set.add(best_line)
    
    return selected


def _local_search_lines(points, lines, max_iters=2000):
    n = len(points)
    m = len(lines)
    current_lines = list(lines)
    current_total = _count_incidences(points, current_lines)
    
    # Precompute incidence counts for current lines
    def line_count(line):
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        return int(np.sum(vals == 0))
    
    # Build candidate set
    candidate_lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            line = _line_from_points(points[i], points[j])
            if line != (0, 0, 0):
                candidate_lines.add(line)
    for A in range(-4, 5):
        for B in range(-4, 5):
            if A == 0 and B == 0:
                continue
            for C in range(-20, 21):
                candidate_lines.add(_normalize_line(A, B, C))
    
    candidates = [l for l in candidate_lines if line_count(l) >= 2]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(m):
            best_new = current_lines[i]
            best_new_count = line_count(current_lines[i])
            for new_line in candidates:
                if new_line == current_lines[i]:
                    continue
                new_count = line_count(new_line)
                # Net change: remove old, add new
                # But incidences aren't independent - a point can be on multiple lines
                # So we need to compute total properly
                old_total = current_total
                current_lines[i] = new_line
                new_total = _count_incidences(points, current_lines)
                if new_total > old_total:
                    best_new = new_line
                    best_new_count = new_count
                    current_total = new_total
                    improved = True
                    break
                else:
                    current_lines[i] = current_lines[i]  # keep
            if improved:
                break
    
    return current_lines, current_total


def _adjust_points(points, lines, max_iters=1000):
    points = points.copy()
    n = len(points)
    current_total = _count_incidences(points, lines)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            best_pos = points[i].copy()
            best_total = current_total
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    old = points[i].copy()
                    points[i] = old + np.array([dx, dy])
                    new_total = _count_incidences(points, lines)
                    if new_total > best_total:
                        best_total = new_total
                        best_pos = points[i].copy()
                    else:
                        points[i] = old
            if best_total > current_total:
                current_total = best_total
                improved = True
                break
        if improved:
            continue
        # Try larger moves
        for i in range(n):
            best_pos = points[i].copy()
            best_total = current_total
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if abs(dx) <= 3 and abs(dy) <= 3:
                        continue
                    if dx == 0 and dy == 0:
                        continue
                    old = points[i].copy()
                    points[i] = old + np.array([dx, dy])
                    new_total = _count_incidences(points, lines)
                    if new_total > best_total:
                        best_total = new_total
                        best_pos = points[i].copy()
                    else:
                        points[i] = old
            if best_total > current_total:
                current_total = best_total
                improved = True
                break
    
    return points, current_total


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    # Initial grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    initial_lines = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            initial_lines.append((a, -1, b))
    lines = np.array(initial_lines, dtype=np.int64)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_total = _count_incidences(points, lines)
    
    # Greedy line selection
    if time.time() < deadline - 200:
        greedy_lines = _greedy_select_lines(points, 27)
        if len(greedy_lines) == 27:
            gl = np.array(greedy_lines, dtype=np.int64)
            gt = _count_incidences(points, gl)
            if gt > best_total:
                best_lines = gl
                best_total = gt
    
    # Local search lines
    if time.time() < deadline - 100:
        ls_lines, ls_total = _local_search_lines(best_points, best_lines, max_iters=1500)
        if ls_total > best_total:
            best_lines = np.array(ls_lines, dtype=np.int64)
            best_total = ls_total
    
    # Adjust points
    if time.time() < deadline - 50:
        adj_points, adj_total = _adjust_points(best_points, best_lines, max_iters=500)
        if adj_total > best_total:
            best_points = adj_points
            best_total = adj_total
    
    # Final line local search
    if time.time() < deadline - 10:
        ls_lines2, ls_total2 = _local_search_lines(best_points, best_lines, max_iters=1000)
        if ls_total2 > best_total:
            best_lines = np.array(ls_lines2, dtype=np.int64)
            best_total = ls_total2
    
    best_points = best_points.astype(np.int64)
    best_lines = best_lines.astype(np.int64)
    
    # Deduplicate lines
    seen = set()
    unique_lines = []
    for l in best_lines:
        norm = _normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if norm not in seen and norm != (0, 0, 0):
            seen.add(norm)
            unique_lines.append(norm)
    
    # If we lost lines, pad with lines from candidates
    if len(unique_lines) < 27:
        # Add lines from point pairs
        all_lines = set(unique_lines)
        for i in range(len(best_points)):
            for j in range(i + 1, len(best_points)):
                line = _line_from_points(best_points[i], best_points[j])
                if line not in all_lines and line != (0, 0, 0):
                    all_lines.add(line)
                    unique_lines.append(line)
                    if len(unique_lines) == 27:
                        break
            if len(unique_lines) == 27:
                break
    
    best_lines = np.array(unique_lines[:27], dtype=np.int64)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
