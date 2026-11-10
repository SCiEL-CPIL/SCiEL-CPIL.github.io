import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_grid(cols, rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    line_set = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            line_set.add(_normalize_line(A, B, C))
    lines_list = list(line_set)
    if not lines_list:
        return points, np.zeros((27, 3), dtype=np.int64)
    lines_arr = np.array(lines_list, dtype=np.int64)
    A = lines_arr[:, 0:1]
    B = lines_arr[:, 1:2]
    C = lines_arr[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    val = A * x + B * y + C
    counts = np.count_nonzero(val == 0, axis=1)
    top = np.argsort(-counts)[:27]
    return points, lines_arr[top]

def _line_polish(points, lines):
    """Cheap final pass: greedily swap a line for the best unused pair-of-points line."""
    lines = lines.copy()
    A = lines[:, 0:1]; B = lines[:, 1:2]; C = lines[:, 2:3]
    x = points[:, 0:1].T; y = points[:, 1:2].T
    M = (A * x + B * y + C == 0)
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    n = len(points)
    # Precompute all pair lines (normalized) + their incidence count, keep top ~150
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]; x2, y2 = points[j]
            A2 = y2 - y1; B2 = x1 - x2; C2 = x2 * y1 - x1 * y2
            if A2 == 0 and B2 == 0:
                continue
            norm = _normalize_line(A2, B2, C2)
            cnt = int(np.count_nonzero(A2 * points[:, 0] + B2 * points[:, 1] + C2 == 0))
            cand.append((cnt, norm))
    cand.sort(key=lambda t: -t[0])
    cand = cand[:200]
    improved = True
    while improved:
        improved = False
        for li in range(27):
            old_cnt = int(np.count_nonzero(M[li, :]))
            cur_norm = line_norms[li]
            for cnt, norm in cand:
                if norm == cur_norm:
                    continue
                if norm in line_norms:
                    continue
                if cnt > old_cnt:
                    A2, B2, C2 = norm
                    M[li, :] = (A2 * points[:, 0] + B2 * points[:, 1] + C2 == 0)
                    lines[li] = [A2, B2, C2]
                    line_norms[li] = norm
                    old_cnt = cnt
                    improved = True
                    break
    return lines

def _anneal(points, lines, deadline, seed=0):
    points = points.copy()
    lines = lines.copy()
    x = points[:, 0].astype(np.int64)  # (54,)
    y = points[:, 1].astype(np.int64)  # (54,)
    A = lines[:, 0].astype(np.int64)   # (27,)
    B = lines[:, 1].astype(np.int64)
    C = lines[:, 2].astype(np.int64)
    M = (A[:, None] * x[None, :] + B[:, None] * y[None, :] + C[:, None] == 0)  # (27,54)
    score = int(np.sum(M))
    best_score = score
    best_points = points.copy()
    best_lines = lines.copy()

    line_norms = set()
    for i in range(27):
        line_norms.add(_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])))
    point_set = set(map(tuple, points.tolist()))

    rng = np.random.default_rng(seed)
    T = 5.0
    cool = 0.99995
    stagnation = 0
    max_stagnation = 1000

    while time.time() < deadline:
        if rng.random() < 0.7:
            pi = int(rng.integers(0, 54))
            ox, oy = points[pi]
            old_col = int(np.count_nonzero(M[:, pi]))
            moved = False
            for _ in range(15):
                dx = int(rng.integers(-6, 7))
                dy = int(rng.integers(-6, 7))
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if (nx, ny) in point_set:
                    continue
                new_col_arr = A * nx + B * ny + C
                new_col = int(np.count_nonzero(new_col_arr == 0))
                delta = new_col - old_col
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    points[pi] = [nx, ny]
                    M[:, pi] = (new_col_arr == 0)
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                        stagnation = 0
                    else:
                        stagnation += 1
                    moved = True
                    break
            if not moved:
                stagnation += 1
        else:
            li = int(rng.integers(0, 27))
            p1 = int(rng.integers(0, 54))
            p2 = int(rng.integers(0, 54))
            if p1 == p2:
                continue
            x1, y1 = points[p1]
            x2, y2 = points[p2]
            A2 = y2 - y1
            B2 = x1 - x2
            C2 = x2 * y1 - x1 * y2
            if A2 == 0 and B2 == 0:
                continue
            norm = _normalize_line(A2, B2, C2)
            current_norm = _normalize_line(int(lines[li][0]), int(lines[li][1]), int(lines[li][2]))
            if norm == current_norm:
                continue
            if norm in line_norms:
                continue
            new_row_arr = A2 * x + B2 * y + C2
            new_row = int(np.count_nonzero(new_row_arr == 0))
            old_row = int(np.count_nonzero(M[li, :]))
            delta = new_row - old_row
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                lines[li] = norm
                line_norms.discard(current_norm)
                line_norms.add(norm)
                M[li, :] = (new_row_arr == 0)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
                    stagnation = 0
                else:
                    stagnation += 1
        T *= cool
        if stagnation > max_stagnation:
            T = 3.0
            stagnation = 0
    return best_points, best_lines, best_score

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    ls = set()
    for i in range(27):
        A, B, C = int(lines[i][0]), int(lines[i][1]), int(lines[i][2])
        if A == 0 and B == 0:
            return False
        n = _normalize_line(A, B, C)
        if n in ls:
            return False
        ls.add(n)
    return True

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 580
    configs = [(9, 6), (6, 9), (18, 3), (3, 18), (27, 2), (12, 5), (5, 12), (2, 27)]
    best_score = -1
    best_points = None
    best_lines = None
    initial_candidates = []
    for c, r in configs:
        pts, lns = _build_grid(c, r)
        s = int(np.sum((lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]) == 0))
        initial_candidates.append((s, c, r, pts, lns))
    initial_candidates.sort(key=lambda x: -x[0])
    time_per_config = 150
    if len(initial_candidates) > 1:
        time_per_config = 100
    if len(initial_candidates) > 2:
        time_per_config = 80
    for idx, (s, c, r, pts, lns) in enumerate(initial_candidates[:3]):
        if time.time() > deadline - 20:
            break
        next_configs = len(initial_candidates) - 1 - idx
        time_left_for_this = min(time_per_config, deadline - time.time() - (next_configs * 10))
        if time_left_for_this < 5:
            continue
        anneal_deadline = time.time() + time_left_for_this
        bp, bl, bs = _anneal(pts, lns, anneal_deadline, seed=42 + idx * 1000)
        if bs > best_score:
            best_score = bs
            best_points = bp.copy()
            best_lines = bl.copy()
    if best_points is None:
        s, c, r, pts, lns = initial_candidates[0]
        best_points = pts
        best_lines = lns
        best_score = s
    # Final polish pass (cheap)
    if time.time() < deadline - 5:
        polished = _line_polish(best_points, best_lines)
        s2 = int(np.sum((polished[:, 0:1] * best_points[:, 0:1].T + polished[:, 1:2] * best_points[:, 1:2].T + polished[:, 2:3]) == 0))
        if s2 > best_score:
            best_lines = polished
            best_score = s2
    if not _is_valid(best_points, best_lines):
        pts, lns = _build_grid(9, 6)
        best_points = pts
        best_lines = lns
    return best_points, best_lines
