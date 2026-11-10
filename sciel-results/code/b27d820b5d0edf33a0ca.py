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
            norm = _normalize_line(A, B, C)
            line_set.add(norm)
    lines_list = list(line_set)
    counts = np.array([int(np.count_nonzero(A * points[:, 0] + B * points[:, 1] + C == 0)) for A, B, C in lines_list])
    top = np.argsort(-counts)[:27]
    lines = np.array([lines_list[i] for i in top], dtype=np.int64)
    return points, lines

def _build_grid_with_diagonals(cols, rows):
    """Build a grid and explicitly include both diagonal directions for high-density lines."""
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    n = len(points)
    px = points[:, 0]
    py = points[:, 1]
    
    # Collect all lines with their incidence counts
    line_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            if norm not in line_dict:
                line_dict[norm] = 0
            line_dict[norm] += 1  # Each pair on the line contributes, but we need actual count
    
    # Recompute actual incidence counts
    lines_list = list(line_dict.keys())
    counts = np.array([int(np.count_nonzero(A * px + B * py + C == 0)) for A, B, C in lines_list])
    top = np.argsort(-counts)[:27]
    lines = np.array([lines_list[i] for i in top], dtype=np.int64)
    return points, lines

def _greedy_line_opt(points, lines, max_rounds=3, deadline=None):
    """Iteratively replace each line with the best-incidence line through a point pair (vectorized)."""
    lines = lines.copy()
    n = len(points)
    px = points[:, 0]
    py = points[:, 1]
    
    # Precompute all pair lines with their normalized forms
    pair_lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            A2 = int(py[j] - py[i])
            B2 = int(px[i] - px[j])
            C2 = int(px[j] * py[i] - px[i] * py[j])
            norm = _normalize_line(A2, B2, C2)
            pair_lines[norm] = (A2, B2, C2)
    
    # Precompute incidence counts for all pair lines
    pair_counts = {}
    for norm, (A2, B2, C2) in pair_lines.items():
        pair_counts[norm] = int(np.count_nonzero(A2 * px + B2 * py + C2 == 0))
    
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for li in range(27):
            old_norm = _normalize_line(int(lines[li][0]), int(lines[li][1]), int(lines[li][2]))
            old_count = int(np.count_nonzero(lines[li, 0] * px + lines[li, 1] * py + lines[li, 2] == 0))
            best_count = old_count
            best_line = None
            for norm, (A2, B2, C2) in pair_lines.items():
                if norm == old_norm:
                    continue
                cnt = pair_counts[norm]
                if cnt > best_count:
                    best_count = cnt
                    best_line = norm
            if best_line is not None and best_count > old_count:
                lines[li] = best_line
                improved = True
        if not improved:
            break
    return lines

def _anneal(points, lines, deadline, seed=0):
    points = points.copy()
    lines = lines.copy()
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    M = (A * x + B * y + C == 0)
    score = int(np.sum(M))
    best_score = score
    best_points = points.copy()
    best_lines = lines.copy()
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    rng = np.random.default_rng(seed)
    T = 3.0
    cool = 0.99997
    point_set = set(map(tuple, points.tolist()))
    
    while time.time() < deadline:
        r = rng.random()
        if r < 0.5:
            # Point move
            pi = int(rng.integers(0, 54))
            ox, oy = points[pi]
            for _ in range(10):
                dx = int(rng.integers(-6, 7))
                dy = int(rng.integers(-6, 7))
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if (nx, ny) in point_set:
                    continue
                new_col_arr = (A * nx + B * ny + C).ravel()
                new_col = int(np.count_nonzero(new_col_arr == 0))
                old_col = int(np.count_nonzero(M[:, pi]))
                delta = new_col - old_col
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    points[pi] = [nx, ny]
                    M[:, pi] = (new_col_arr == 0).astype(bool)
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break
        elif r < 0.7:
            # Line replacement: pick line through two random points
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
            dup = False
            for i in range(27):
                if i != li and line_norms[i] == norm:
                    dup = True
                    break
            if dup:
                continue
            new_row_arr = (A2 * points[:, 0] + B2 * points[:, 1] + C2)
            new_row = int(np.count_nonzero(new_row_arr == 0))
            old_row = int(np.count_nonzero(M[li, :]))
            delta = new_row - old_row
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                lines[li] = norm
                line_norms[li] = norm
                M[li, :] = (new_row_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        elif r < 0.85:
            # Point-to-line snap: move a point onto a random existing line
            pi = int(rng.integers(0, 54))
            li = int(rng.integers(0, 27))
            A_l, B_l, C_l = int(lines[li][0]), int(lines[li][1]), int(lines[li][2])
            ox, oy = int(points[pi][0]), int(points[pi][1])
            candidates = []
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    nx, ny = ox + dx, oy + dy
                    if A_l * nx + B_l * ny + C_l != 0:
                        continue
                    if (nx, ny) == (ox, oy):
                        continue
                    if (nx, ny) in point_set:
                        continue
                    candidates.append((nx, ny))
            if not candidates:
                continue
            nx, ny = candidates[int(rng.integers(0, len(candidates)))]
            new_col_arr = (A * nx + B * ny + C).ravel()
            new_col = int(np.count_nonzero(new_col_arr == 0))
            old_col = int(np.count_nonzero(M[:, pi]))
            delta = new_col - old_col
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                point_set.discard((ox, oy))
                point_set.add((nx, ny))
                points[pi] = [nx, ny]
                M[:, pi] = (new_col_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        else:
            # Swap two points
            p1 = int(rng.integers(0, 54))
            p2 = int(rng.integers(0, 54))
            if p1 == p2:
                continue
            ox1, oy1 = points[p1]
            ox2, oy2 = points[p2]
            # Try swapping positions
            new_pos1 = (int(ox2), int(oy2))
            new_pos2 = (int(ox1), int(oy1))
            if new_pos1 == new_pos2:
                continue
            # Check if positions are valid (not occupied by other points)
            if new_pos1 in point_set and new_pos1 != (ox1, oy1):
                continue
            if new_pos2 in point_set and new_pos2 != (ox2, oy2):
                continue
            # Compute delta
            old_col1 = int(np.count_nonzero(M[:, p1]))
            old_col2 = int(np.count_nonzero(M[:, p2]))
            new_col1_arr = (A * int(ox2) + B * int(oy2) + C).ravel()
            new_col2_arr = (A * int(ox1) + B * int(oy1) + C).ravel()
            new_col1 = int(np.count_nonzero(new_col1_arr == 0))
            new_col2 = int(np.count_nonzero(new_col2_arr == 0))
            delta = (new_col1 + new_col2) - (old_col1 + old_col2)
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                point_set.discard((ox1, oy1))
                point_set.discard((ox2, oy2))
                point_set.add(new_pos1)
                point_set.add(new_pos2)
                points[p1] = [int(ox2), int(oy2)]
                points[p2] = [int(ox1), int(oy1)]
                M[:, p1] = (new_col1_arr == 0).astype(bool)
                M[:, p2] = (new_col2_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        T *= cool
    return best_points, best_lines, best_score

def _polish(points, lines, deadline, seed=1000):
    """Final polish: greedy line optimization + short anneal with low temperature + point-to-line snaps."""
    points = points.copy()
    lines = lines.copy()
    
    # Greedy line optimization
    lines = _greedy_line_opt(points, lines, max_rounds=5, deadline=deadline)
    
    # Recompute
    M = (lines[:, 0:1] * points[:, 0:1].T + lines[:, 1:2] * points[:, 1:2].T + lines[:, 2:3] == 0)
    score = int(np.sum(M))
    best_score = score
    best_points = points.copy()
    best_lines = lines.copy()
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    
    # Short anneal with low temperature
    rng = np.random.default_rng(seed)
    T = 1.0
    cool = 0.9999
    point_set = set(map(tuple, points.tolist()))
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    
    while time.time() < deadline:
        r = rng.random()
        if r < 0.35:
            pi = int(rng.integers(0, 54))
            ox, oy = points[pi]
            for _ in range(8):
                dx = int(rng.integers(-5, 6))
                dy = int(rng.integers(-5, 6))
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if (nx, ny) in point_set:
                    continue
                new_col_arr = (A * nx + B * ny + C).ravel()
                new_col = int(np.count_nonzero(new_col_arr == 0))
                old_col = int(np.count_nonzero(M[:, pi]))
                delta = new_col - old_col
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    points[pi] = [nx, ny]
                    M[:, pi] = (new_col_arr == 0).astype(bool)
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break
        elif r < 0.6:
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
            dup = False
            for i in range(27):
                if i != li and line_norms[i] == norm:
                    dup = True
                    break
            if dup:
                continue
            new_row_arr = (A2 * points[:, 0] + B2 * points[:, 1] + C2)
            new_row = int(np.count_nonzero(new_row_arr == 0))
            old_row = int(np.count_nonzero(M[li, :]))
            delta = new_row - old_row
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                lines[li] = norm
                line_norms[li] = norm
                M[li, :] = (new_row_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        else:
            # Point-to-line snap
            pi = int(rng.integers(0, 54))
            li = int(rng.integers(0, 27))
            A_l, B_l, C_l = int(lines[li][0]), int(lines[li][1]), int(lines[li][2])
            ox, oy = int(points[pi][0]), int(points[pi][1])
            candidates = []
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    nx, ny = ox + dx, oy + dy
                    if A_l * nx + B_l * ny + C_l != 0:
                        continue
                    if (nx, ny) == (ox, oy):
                        continue
                    if (nx, ny) in point_set:
                        continue
                    candidates.append((nx, ny))
            if not candidates:
                continue
            nx, ny = candidates[int(rng.integers(0, len(candidates)))]
            new_col_arr = (A * nx + B * ny + C).ravel()
            new_col = int(np.count_nonzero(new_col_arr == 0))
            old_col = int(np.count_nonzero(M[:, pi]))
            delta = new_col - old_col
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                point_set.discard((ox, oy))
                point_set.add((nx, ny))
                points[pi] = [nx, ny]
                M[:, pi] = (new_col_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        T *= cool
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
    deadline = start + 590
    configs = [(6, 9), (9, 6), (3, 18), (2, 27), (18, 3), (27, 2)]
    best_score = -1
    best_points = None
    best_lines = None
    
    # Phase 1: Build grids and greedy-optimize lines
    for idx, (c, r) in enumerate(configs):
        if time.time() > deadline - 120:
            break
        pts, lns = _build_grid(c, r)
        # Greedy line optimization
        lns = _greedy_line_opt(pts, lns, max_rounds=3, deadline=deadline)
        s = int(np.sum((lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]) == 0))
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            best_lines = lns.copy()
    
    # Phase 2: Multiple anneal restarts with different seeds
    if best_points is not None and time.time() < deadline - 90:
        anneal_seeds = [42, 123, 999]
        time_per_anneal = (deadline - 90 - time.time()) / len(anneal_seeds)
        for seed in anneal_seeds:
            if time.time() > deadline - 30:
                break
            time_left = time.time() + time_per_anneal
            bp, bl, bs = _anneal(best_points, best_lines, time_left, seed=seed)
            if bs > best_score:
                best_score = bs
                best_points = bp.copy()
                best_lines = bl.copy()
    
    # Phase 3: Polish
    if best_points is not None and time.time() < deadline - 20:
        time_left = deadline - 10
        bp, bl, bs = _polish(best_points, best_lines, time_left, seed=777)
        if bs > best_score:
            best_score = bs
            best_points = bp.copy()
            best_lines = bl.copy()
    
    if best_points is None:
        best_points, best_lines = _build_grid(6, 9)
    if not _is_valid(best_points, best_lines):
        best_points, best_lines = _build_grid(6, 9)
    return best_points, best_lines
