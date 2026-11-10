# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from math import gcd


def _gcd(a, b, c):
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g if g != 0 else 1


def _normalize_line(A, B, C):
    g = _gcd(A, B, C)
    A //= g
    B //= g
    C //= g
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def _count_incidents(points, lines):
    vals = lines[:, 0][None, :] * points[:, 0:1] + lines[:, 1][None, :] * points[:, 1:2] + lines[:, 2][None, :]
    return int(np.sum(vals == 0))


def _validate(points, lines):
    """Check all validity constraints. Return True if valid."""
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.allclose(points, np.round(points), atol=1e-6) or not np.allclose(lines, np.round(lines), atol=1e-6):
        return False
    # Repeated points
    pt_set = set()
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t in pt_set:
            return False
        pt_set.add(t)
    # Lines
    line_set = set()
    for l in lines:
        A, B, C = int(l[0]), int(l[1]), int(l[2])
        if A == 0 and B == 0:
            return False
        nl = _normalize_line(A, B, C)
        if nl in line_set:
            return False
        line_set.add(nl)
    return True


def _line_through_two_points(p1, p2):
    """Return normalized (A,B,C) for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _build_grid_base():
    """6x9 grid with 15 grid lines + 12 best diagonals."""
    points_x = np.arange(6)
    points_y = np.arange(9)
    px, py = np.meshgrid(points_x, points_y, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for i in range(6):
        lines.append((1, 0, -i))
    for j in range(9):
        lines.append((0, 1, -j))

    # Generate candidates for diagonals
    candidates = []
    for c in range(-12, 12):
        candidates.append((1, -1, c))
    for c in range(-12, 18):
        candidates.append((1, 1, c))
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            if A == 1 and B == -1:
                continue
            if A == 1 and B == 1:
                continue
            for C in range(-25, 25):
                candidates.append((A, B, C))

    unique_lines = set()
    for l in candidates:
        unique_lines.add(_normalize_line(*l))

    scored_lines = []
    for l in unique_lines:
        A, B, C = l
        vals = A * points[:, 0] + B * points[:, 1] + C
        cnt = int(np.sum(vals == 0))
        if cnt > 0:
            scored_lines.append((cnt, l))

    scored_lines.sort(key=lambda x: -x[0])

    grid_set = set([(1, 0, -i) for i in range(6)] + [(0, 1, -j) for j in range(9)])
    selected_diags = []
    for cnt, l in scored_lines:
        if l not in grid_set and len(selected_diags) < 12:
            selected_diags.append(l)

    for l in selected_diags:
        lines.append(l)

    lines = np.array(lines, dtype=np.int64)
    return points, lines


def _fast_joint_search(points, lines, max_time):
    """
    Joint local search: alternate between
    (1) swapping the worst line for the best non-selected line through point pairs, and
    (2) moving individual points to better positions.
    Returns (best_points, best_lines, best_count).
    """
    n = 54
    m = 27
    deadline = time.time() + max_time

    best_points = points.copy()
    best_lines = [tuple(l) for l in lines]
    best_count = _count_incidents(best_points, np.array(best_lines, dtype=np.int64))

    # Precompute line coefficients
    A_arr = np.array([l[0] for l in best_lines], dtype=np.int64)
    B_arr = np.array([l[1] for l in best_lines], dtype=np.int64)
    C_arr = np.array([l[2] for l in best_lines], dtype=np.int64)

    # Candidate point positions (expanded range)
    x_cand = np.arange(-8, 16, dtype=np.int64)
    y_cand = np.arange(-8, 22, dtype=np.int64)

    # Generate all candidate lines from point pairs (recomputed as points move)
    def _generate_candidate_lines(pts):
        cand = {}
        for i in range(n):
            for j in range(i + 1, n):
                nl = _line_through_two_points(pts[i], pts[j])
                if nl not in cand:
                    A, B, C = nl
                    vals = A * pts[:, 0] + B * pts[:, 1] + C
                    cand[nl] = int(np.sum(vals == 0))
        return cand

    def _line_count(nl, pts):
        A, B, C = nl
        vals = A * pts[:, 0] + B * pts[:, 1] + C
        return int(np.sum(vals == 0))

    def _point_incidence_count(idx, pts, A_arr, B_arr, C_arr):
        x, y = int(pts[idx, 0]), int(pts[idx, 1])
        return int(np.sum(A_arr * x + B_arr * y + C_arr == 0))

    no_improve = 0
    max_no_improve = 60
    rng = np.random.default_rng(123)

    while time.time() < deadline - 3 and no_improve < max_no_improve:
        improved = False

        # --- Phase A: Line swap ---
        if time.time() > deadline - 3:
            break

        # Recompute candidate lines from current points
        cand_lines = _generate_candidate_lines(best_points)
        current_set = set(best_lines)

        # Find worst line
        worst_idx = 0
        worst_cnt = _line_count(best_lines[0], best_points)
        for idx in range(1, m):
            c = _line_count(best_lines[idx], best_points)
            if c < worst_cnt:
                worst_cnt = c
                worst_idx = idx

        # Find best replacement
        best_gain = 0
        best_repl = None
        for nl, cnt in cand_lines.items():
            if nl in current_set:
                continue
            gain = cnt - worst_cnt
            if gain > best_gain:
                best_gain = gain
                best_repl = nl

        if best_repl is not None and best_gain > 0:
            best_lines[worst_idx] = best_repl
            A_arr[worst_idx] = best_repl[0]
            B_arr[worst_idx] = best_repl[1]
            C_arr[worst_idx] = best_repl[2]
            best_count += best_gain
            improved = True
            no_improve = 0
            continue  # Re-enter loop for point moves

        # --- Phase B: Point moves ---
        if time.time() > deadline - 3:
            break

        # Try moving each point to a better position
        for i in range(n):
            if time.time() > deadline - 3:
                break

            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            old_on = _point_incidence_count(i, best_points, A_arr, B_arr, C_arr)

            # Try candidate positions
            best_new_on = old_on
            best_cx, best_cy = orig_x, orig_y
            best_gain_pt = 0

            # Try a subset of candidates for speed
            for cx in x_cand:
                for cy in y_cand:
                    if cx == orig_x and cy == orig_y:
                        continue
                    # Check collision
                    collision = False
                    for j in range(n):
                        if j != i and int(best_points[j, 0]) == cx and int(best_points[j, 1]) == cy:
                            collision = True
                            break
                    if collision:
                        continue
                    new_on = int(np.sum(A_arr * cx + B_arr * cy + C_arr == 0))
                    gain = new_on - old_on
                    if gain > best_gain_pt:
                        best_gain_pt = gain
                        best_new_on = new_on
                        best_cx, best_cy = cx, cy

            if best_gain_pt > 0:
                best_points[i, 0] = best_cx
                best_points[i, 1] = best_cy
                best_count += best_gain_pt
                improved = True
                no_improve = 0

        if not improved:
            no_improve += 1

    return best_points, np.array(best_lines, dtype=np.int64), best_count


def _fast_line_optimize(points, max_time):
    """Optimize lines for fixed points using pair-based candidates and greedy + swap."""
    n = 54
    m = 27
    deadline = time.time() + max_time

    # Generate all candidate lines from point pairs
    cand_lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            nl = _line_through_two_points(points[i], points[j])
            if nl not in cand_lines:
                A, B, C = nl
                vals = A * points[:, 0] + B * points[:, 1] + C
                cand_lines[nl] = int(np.sum(vals == 0))

    # Sort by count
    sorted_lines = sorted(cand_lines.items(), key=lambda x: -x[1])

    # Greedy select 27
    selected = []
    selected_set = set()
    for nl, cnt in sorted_lines:
        if nl in selected_set:
            continue
        selected.append(nl)
        selected_set.add(nl)
        if len(selected) == m:
            break

    # Pad if needed
    if len(selected) < m:
        for c in range(-100, 100):
            nl = _normalize_line(0, 1, c)
            if nl not in selected_set:
                selected.append(nl)
                selected_set.add(nl)
                if len(selected) == m:
                    break
        if len(selected) < m:
            for c in range(-100, 100):
                nl = _normalize_line(1, 0, c)
                if nl not in selected_set:
                    selected.append(nl)
                    selected_set.add(nl)
                    if len(selected) == m:
                        break

    lines_arr = np.array(selected, dtype=np.int64)
    count = _count_incidents(points, lines_arr)

    # Local search: swap worst for best non-selected
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        current_set = set(selected)

        # Find worst
        worst_idx = 0
        worst_cnt = _line_count(selected[0], points)
        for idx in range(1, m):
            c = _line_count(selected[idx], points)
            if c < worst_cnt:
                worst_cnt = c
                worst_idx = idx

        # Find best replacement
        best_gain = 0
        best_repl = None
        for nl, cnt in sorted_lines:
            if nl in current_set:
                continue
            gain = cnt - worst_cnt
            if gain > best_gain:
                best_gain = gain
                best_repl = nl

        if best_repl is not None and best_gain > 0:
            current_set.discard(selected[worst_idx])
            selected[worst_idx] = best_repl
            current_set.add(best_repl)
            lines_arr = np.array(selected, dtype=np.int64)
            count += best_gain
            improved = True

    return lines_arr, count


def _point_hill_climb(points, lines, max_time):
    """Move points to maximize incidences with fixed lines."""
    n = 54
    m = 27
    deadline = time.time() + max_time

    best_points = points.copy()
    best_count = _count_incidents(best_points, lines)

    A_arr = lines[:, 0].astype(np.int64)
    B_arr = lines[:, 1].astype(np.int64)
    C_arr = lines[:, 2].astype(np.int64)

    x_cand = np.arange(-8, 16, dtype=np.int64)
    y_cand = np.arange(-8, 22, dtype=np.int64)

    no_improve = 0
    max_no_improve = 80

    while time.time() < deadline - 3 and no_improve < max_no_improve:
        improved = False
        for i in range(n):
            if time.time() > deadline - 3:
                break
            orig_x, orig_y = int(best_points[i, 0]), int(best_points[i, 1])
            old_on = int(np.sum(A_arr * orig_x + B_arr * orig_y + C_arr == 0))

            best_gain = 0
            best_cx, best_cy = orig_x, orig_y

            for cx in x_cand:
                for cy in y_cand:
                    if cx == orig_x and cy == orig_y:
                        continue
                    collision = False
                    for j in range(n):
                        if j != i and int(best_points[j, 0]) == cx and int(best_points[j, 1]) == cy:
                            collision = True
                            break
                    if collision:
                        continue
                    new_on = int(np.sum(A_arr * cx + B_arr * cy + C_arr == 0))
                    gain = new_on - old_on
                    if gain > best_gain:
                        best_gain = gain
                        best_cx, best_cy = cx, cy

            if best_gain > 0:
                best_points[i, 0] = best_cx
                best_points[i, 1] = best_cy
                best_count += best_gain
                improved = True
                no_improve = 0

        if not improved:
            no_improve += 1

    return best_points, best_count


def szemeredi_trotter_incidences():
    start_time = time.time()
    total_budget = 590  # 630s limit, 40s safety margin

    # Start with grid base
    points, lines = _build_grid_base()
    best_count = _count_incidents(points, lines)

    # Phase 1: Optimize lines for grid points
    if time.time() - start_time < total_budget - 60:
        lines_opt, count_opt = _fast_line_optimize(points, max_time=90)
        if count_opt > best_count:
            lines = lines_opt
            best_count = count_opt

    # Phase 2: Joint search (points + lines)
    if time.time() - start_time < total_budget - 30:
        remaining = total_budget - (time.time() - start_time) - 10
        if remaining > 30:
            pts2, lines2, cnt2 = _fast_joint_search(points, lines, max_time=remaining)
            if cnt2 > best_count:
                points = pts2
                lines = lines2
                best_count = cnt2

    # Phase 3: Another joint search round
    if time.time() - start_time < total_budget - 15:
        remaining = total_budget - (time.time() - start_time) - 5
        if remaining > 20:
            pts3, lines3, cnt3 = _fast_joint_search(points, lines, max_time=remaining)
            if cnt3 > best_count:
                points = pts3
                lines = lines3
                best_count = cnt3

    # Final validation
    if not _validate(points, lines):
        points, lines = _build_grid_base()

    return points, lines


# EVOLVE-BLOCK-END
