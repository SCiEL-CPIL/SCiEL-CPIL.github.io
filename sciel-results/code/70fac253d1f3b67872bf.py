# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from math import gcd

_T0 = time.time()
_DEADLINE = 620.0  # safety margin vs 630s kill


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _canon(A, B, C):
    """Canonical form: divide by gcd, make first nonzero coefficient positive."""
    g = gcd(abs(int(A)), gcd(abs(int(B)), abs(int(C))))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _evaluate(points, lines):
    """Exact incidence count in int64."""
    total = 0
    for (px, py) in points:
        for (A, B, C) in lines:
            if A * px + B * py + C == 0:
                total += 1
    return total


def _compute_all_lines(points):
    """Compute all distinct lines through pairs of points with their incidence counts."""
    P = list(points)
    n = len(P)
    idx = np.array(P, dtype=np.int64)
    line_info = {}  # canon -> (A_, B_, C_, cnt, point_indices)
    
    for i in range(n):
        x1, y1 = P[i]
        for j in range(i + 1, n):
            x2, y2 = P[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            ct = _canon(A, B, C)
            if ct not in line_info:
                A_, B_, C_ = ct
                mask = (A_ * idx[:, 0] + B_ * idx[:, 1] + C_ == 0)
                cnt = int(np.sum(mask))
                pts_on_line = np.where(mask)[0]
                line_info[ct] = (A_, B_, C_, cnt, pts_on_line)
    return line_info


def _greedy_select(line_info, n_lines=27):
    """Greedy: pick top-n_lines by incidence count."""
    cand_list = sorted(line_info.items(), key=lambda kv: -kv[1][3])
    chosen = []
    total = 0
    for ct, (A_, B_, C_, cnt, pts) in cand_list:
        if len(chosen) >= n_lines:
            break
        chosen.append(ct)
        total += cnt
    return chosen, total


def _local_search_lines(line_info, initial_chosen, n_lines=27, max_iters=200):
    """Local search: try swapping one chosen line with one unchosen line if it improves total."""
    chosen = list(initial_chosen)
    chosen_set = set(chosen)
    
    # Precompute incidence for each line
    inc = {ct: line_info[ct][3] for ct in chosen}
    total = sum(inc.values())
    
    # Sort unchosen lines by incidence for faster search
    unchosen = [(ct, line_info[ct][3]) for ct in line_info if ct not in chosen_set]
    unchosen.sort(key=lambda x: -x[1])
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if _time_left() < 10:
            break
        # Try each unchosen line as replacement for each chosen line
        for new_ct, new_inc in unchosen:
            if new_ct in chosen_set:
                continue
            for old_idx, old_ct in enumerate(chosen):
                gain = new_inc - inc[old_ct]
                if gain > 0:
                    # Accept the swap
                    chosen[old_idx] = new_ct
                    chosen_set.discard(old_ct)
                    chosen_set.add(new_ct)
                    del inc[old_ct]
                    inc[new_ct] = new_inc
                    total += gain
                    improved = True
                    break  # Restart scan with updated state
            if improved:
                break
    return chosen, total


def _solve_one(points, n_lines=27):
    """Solve: compute all lines, greedy select, then local search."""
    line_info = _compute_all_lines(points)
    if not line_info:
        return [], 0
    
    # Greedy initial selection
    chosen, total = _greedy_select(line_info, n_lines)
    
    # Local search to improve
    chosen, total = _local_search_lines(line_info, chosen, n_lines)
    
    # Build result
    chosen_coef = [line_info[ct][:3] for ct in chosen]
    return chosen_coef, total


def _build_points_grid(w, h, n_pts=54, rng=None):
    """Build point set from w x h grid, optionally removing random points."""
    if rng is None:
        rng = random
    all_pts = [(x, y) for x in range(w) for y in range(h)]
    if len(all_pts) <= n_pts:
        return all_pts
    return rng.sample(all_pts, n_pts)


def _build_points_structured(w, h, n_pts=54, rng=None):
    """Try structured point sets: full grid if fits, else strategic removals."""
    if rng is None:
        rng = random
    all_pts = [(x, y) for x in range(w) for y in range(h)]
    if len(all_pts) == n_pts:
        return all_pts
    if len(all_pts) < n_pts:
        return all_pts
    # Remove points from corners/edges first, then random
    corners = [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]
    edge_pts = [(x,y) for x,y in all_pts if x in (0,w-1) or y in (0,h-1)]
    interior = [(x,y) for x,y in all_pts if x not in (0,w-1) and y not in (0,h-1)]
    
    to_remove = len(all_pts) - n_pts
    # Try removing from corners first, then edges, then random interior
    removal_candidates = corners + [p for p in edge_pts if p not in corners] + rng.sample(interior, min(len(interior), to_remove))
    rng.shuffle(removal_candidates)
    to_remove_set = set(removal_candidates[:to_remove])
    result = [p for p in all_pts if p not in to_remove_set]
    if len(result) != n_pts:
        # Fallback to random
        return rng.sample(all_pts, n_pts)
    return result


def _try_point_set(points, current_best):
    """Try a point set, return (lines, total) if it beats current best."""
    lines, total = _solve_one(points)
    if total > current_best[0]:
        return lines, total
    return None, current_best[0]


def _run_search():
    global _best_score, _best_points, _best_lines
    rng = random.Random(12345)
    
    # Seed baseline first
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    seed_pts = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64).tolist()
    rows = [(a, -1, b) for a in range(1, k + 1) for b in range(1, k * k + 1)]
    seed_total = _evaluate(seed_pts, rows)
    _best_score = seed_total
    _best_points = [tuple(int(v) for v in p) for p in seed_pts]
    _best_lines = [tuple(int(v) for v in l) for l in rows]
    
    # Try various grid configurations
    # Focus on grids where w*h is close to 54 for good density
    configs = [
        # (w, h, mode) where mode: 'full' if w*h==54, 'remove' if w*h>54
        (6, 9, 'full'),
        (7, 8, 'remove'),
        (9, 6, 'full'),
        (8, 7, 'remove'),
        (5, 11, 'remove'),
        (11, 5, 'remove'),
        (4, 14, 'remove'),
        (14, 4, 'remove'),
        (3, 18, 'remove'),
        (18, 3, 'remove'),
        (2, 27, 'remove'),
        (27, 2, 'remove'),
        (1, 54, 'remove'),
        (54, 1, 'remove'),
        (10, 6, 'remove'),
        (6, 10, 'remove'),
        (12, 5, 'remove'),
        (5, 12, 'remove'),
        (9, 7, 'remove'),
        (7, 9, 'remove'),
        (13, 5, 'remove'),
        (5, 13, 'remove'),
    ]
    
    for (w, h, mode) in configs:
        if _time_left() < 30:
            break
        
        if mode == 'full' and w * h == 54:
            # Try full grid
            points = _build_points_grid(w, h, 54, rng)
            lines, total = _try_point_set(points, (_best_score, None))
            if lines is not None:
                _best_score = total
                _best_points = [tuple(int(v) for v in p) for p in points]
                _best_lines = [tuple(int(v) for v in l) for l in lines]
        
        # Try structured removals and random removals
        n_trials = 15 if w * h <= 80 else 8
        for t in range(n_trials):
            if _time_left() < 15:
                break
            if t < n_trials // 3:
                points = _build_points_structured(w, h, 54, rng)
            else:
                points = _build_points_grid(w, h, 54, rng)
            lines, total = _try_point_set(points, (_best_score, None))
            if lines is not None:
                _best_score = total
                _best_points = [tuple(int(v) for v in p) for p in points]
                _best_lines = [tuple(int(v) for v in l) for l in lines]
    
    # If time remains, do more intensive search on best-performing configs
    if _time_left() > 60:
        for (w, h, mode) in [(6, 9, 'full'), (7, 8, 'remove'), (9, 6, 'full'), (8, 7, 'remove')]:
            if _time_left() < 20:
                break
            for t in range(20):
                if _time_left() < 10:
                    break
                points = _build_points_grid(w, h, 54, rng)
                lines, total = _try_point_set(points, (_best_score, None))
                if lines is not None:
                    _best_score = total
                    _best_points = [tuple(int(v) for v in p) for p in points]
                    _best_lines = [tuple(int(v) for v in l) for l in lines]


_best_score = 0
_best_points = None
_best_lines = None


def _finalize():
    """Validate and return best (points, lines) found, or fall back to seed."""
    global _best_points, _best_lines, _best_score
    if _best_points is not None and _best_lines is not None:
        pts = np.array(_best_points, dtype=np.int64)
        lns = np.array(_best_lines, dtype=np.int64)
        # Basic sanity checks
        if pts.shape == (54, 2) and lns.shape == (27, 3):
            # verify no A=B=0
            if not np.any((lns[:, 0] == 0) & (lns[:, 1] == 0)):
                return pts, lns
    # Fallback: seed construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    return points, lines


def szemeredi_trotter_incidences():
    """
    Returns (points, lines): points a (54,2) int64 array, lines a (27,3) int64 array.
    Maximizes point-line incidences via greedy + local search line selection over grid point sets.
    """
    _run_search()
    return _finalize()


# EVOLVE-BLOCK-END
