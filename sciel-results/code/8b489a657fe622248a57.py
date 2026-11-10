# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd


def _canon(A, B, C):
    """Canonical form: divide by gcd, make first nonzero coefficient positive."""
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and (B < 0 or (B == 0 and C < 0))):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_two_points(p1, p2):
    """Return (A,B,C) canonical int tuple for line through p1=(x1,y1), p2=(x2,y2)."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _canon(A, B, C)


def _incidence_count(points, lines):
    """Count incidences exactly in int64."""
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    vals = L[:, 0:1] * P[:, 0:1].T + L[:, 1:2] * P[:, 1:2].T + L[:, 2:3]
    return int(np.count_nonzero(vals == 0))


def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.issubdtype(points.dtype, np.integer):
        return False
    if not np.issubdtype(lines.dtype, np.integer):
        return False
    if len(np.unique(points, axis=0)) != 54:
        return False
    seen = set()
    for row in lines:
        A, B, C = int(row[0]), int(row[1]), int(row[2])
        if A == 0 and B == 0:
            return False
        key = _canon(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True


def _grid_construction():
    """Classical Szemeredi-Trotter grid: 54 points, 27 lines, 81 incidences."""
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


def _star_construction():
    """
    Alternative: place points on a small grid and lines through pairs.
    Try a 6x9 grid of points (54) and pick 27 lines that maximize incidences.
    """
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n = len(points)
    line_scores = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            line_scores[key] = line_scores.get(key, 0) + 1
    sorted_keys = sorted(line_scores.items(), key=lambda x: -x[1])
    lines = []
    seen = set()
    for key, _ in sorted_keys:
        if key in seen:
            continue
        lines.append(key)
        seen.add(key)
        if len(lines) == 27:
            break
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _random_grid_construction(w, h, rng):
    """Pick 54 distinct points from a w x h grid, then greedy top-27 lines."""
    all_pts = [(x, y) for x in range(w) for y in range(h)]
    if len(all_pts) < 54:
        return None
    # rng.choice works with lists/arrays of hashes. It returns an array of the chosen items.
    # We need to ensure we get 54 unique items.
    idxs = rng.choice(len(all_pts), size=54, replace=False)
    pts = [all_pts[i] for i in idxs]
    points = np.array(pts, dtype=np.int64)
    n = len(points)
    line_scores = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_two_points(points[i], points[j])
            line_scores[key] = line_scores.get(key, 0) + 1
    sorted_keys = sorted(line_scores.items(), key=lambda x: -x[1])
    lines = []
    seen = set()
    for key, _ in sorted_keys:
        if key in seen:
            continue
        lines.append(key)
        seen.add(key)
        if len(lines) == 27:
            break
    if len(lines) < 27:
        return None
    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _local_search(points, lines, max_iters=5000, deadline=None, rng=None):
    """
    Local search: try replacing a point or a line to increase incidences.
    Uses vectorized incidence counting for speed.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_pts = len(points)
    n_lines = len(lines)
    best_inc = _incidence_count(points, lines)
    P = points.copy()
    L = lines.copy()
    
    # Precompute current line set for quick duplicate check
    current_line_set = set()
    for row in L:
        current_line_set.add(_canon(int(row[0]), int(row[1]), int(row[2])))
    
    # Precompute all candidate lines through pairs of current points
    # This is expensive but done once. For 54 points, ~1431 pairs.
    candidate_lines = {}
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            key = _line_from_two_points(P[i], P[j])
            candidate_lines[key] = True
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try moving a random point to a nearby integer location
        idx = rng.integers(0, n_pts)
        old_pt = P[idx].copy()
        found_move = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_pt = old_pt + np.array([dx, dy])
                # Check no duplicate points: ensure new_pt is not equal to any other point
                # We can check if new_pt exists in P excluding idx
                # Using a simple loop or vectorized comparison
                # P is (N, 2). Compare new_pt to all rows.
                # If any row (other than idx) equals new_pt, it's a duplicate.
                matches = np.all(P == new_pt, axis=1)
                if matches.sum() > 0:
                    # If it matches, check if it's only matching itself (which is now set to new_pt)
                    # Actually, if we set P[idx] = new_pt, and new_pt was already at P[k] (k!=idx),
                    # then matches will have True at idx and k. So sum >= 2.
                    # If new_pt was NOT in P before, matches will have True only at idx (sum=1).
                    # So if sum > 1, it's a duplicate.
                    if matches.sum() > 1:
                        P[idx] = old_pt
                        continue
                P[idx] = new_pt
                inc = _incidence_count(P, L)
                if inc > best_inc:
                    best_inc = inc
                    improved = True
                    found_move = True
                    # Update candidate lines (approximate: just mark as dirty)
                    # For simplicity, we don't fully update candidate_lines here
                    break
                P[idx] = old_pt
            if found_move:
                break
        
        if improved:
            continue
        
        # Try replacing a random line with a line through two random points
        lidx = rng.integers(0, n_lines)
        old_line = L[lidx].copy()
        # Pick from candidate lines
        cand_list = list(candidate_lines.keys())
        if cand_list:
            for _ in range(5):
                new_line = cand_list[rng.integers(0, len(cand_list))]
                # Check no duplicate lines (except the one we're replacing)
                dup = False
                for k in range(n_lines):
                    if k == lidx:
                        continue
                    if _canon(int(L[k, 0]), int(L[k, 1]), int(L[k, 2])) == new_line:
                        dup = True
                        break
                if dup:
                    continue
                L[lidx] = new_line
                inc = _incidence_count(P, L)
                if inc > best_inc:
                    best_inc = inc
                    improved = True
                    break
                L[lidx] = old_line
        
        if improved:
            continue
        
        # Try swapping two points
        if rng.random() < 0.3:
            i = rng.integers(0, n_pts)
            j = rng.integers(0, n_pts)
            if i == j:
                continue
            P[[i, j]] = P[[j, i]]
            inc = _incidence_count(P, L)
            if inc > best_inc:
                best_inc = inc
                improved = True
            else:
                P[[i, j]] = P[[j, i]]
    
    return P, L, best_inc


def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 600  # safety margin before 630s kill
    rng = np.random.default_rng(12345)
    
    # Start with grid construction
    points, lines = _grid_construction()
    best_inc = _incidence_count(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()
    
    # Try star construction
    pts2, lns2 = _star_construction()
    inc2 = _incidence_count(pts2, lns2)
    if inc2 > best_inc:
        best_inc = inc2
        best_points = pts2.copy()
        best_lines = lns2.copy()
    
    # Try random grid constructions with various aspect ratios
    aspects = [
        (6, 9), (7, 8), (9, 6), (8, 7), (10, 6), (6, 10),
        (5, 12), (12, 5), (4, 14), (14, 4), (5, 11), (11, 5),
        (3, 18), (18, 3), (2, 27), (27, 2), (9, 7), (7, 9),
        (13, 5), (5, 13), (4, 15), (15, 4), (10, 7), (7, 10),
    ]
    
    for (w, h) in aspects:
        if time.time() > deadline - 30:
            break
        # Try a few random samples from this grid
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            result = _random_grid_construction(w, h, rng)
            if result is None:
                continue
            p, l = result
            inc = _incidence_count(p, l)
            if inc > best_inc:
                best_inc = inc
                best_points = p.copy()
                best_lines = l.copy()
    
    # Local search from best starting point
    if time.time() < deadline - 60:
        P, L, inc = _local_search(best_points, best_lines, max_iters=15000, deadline=deadline - 30, rng=rng)
        if inc > best_inc:
            best_inc = inc
            best_points = P
            best_lines = L
    
    # Final local search round
    if time.time() < deadline - 30:
        P, L, inc = _local_search(best_points, best_lines, max_iters=10000, deadline=deadline - 15, rng=rng)
        if inc > best_inc:
            best_inc = inc
            best_points = P
            best_lines = L
    
    # Ensure validity
    if not _is_valid(best_points, best_lines):
        # Fallback to grid
        best_points, best_lines = _grid_construction()
    
    return best_points.astype(np.int64), best_lines.astype(np.int64)


# EVOLVE-BLOCK-END
