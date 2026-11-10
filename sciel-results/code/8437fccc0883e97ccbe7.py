import numpy as np
from math import gcd
import time


def _gcd3(a, b, c):
    a, b, c = abs(int(a)), abs(int(b)), abs(int(c))
    g = gcd(gcd(a, b), c)
    return g if g > 0 else 1


def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidents(points, lines):
    total = 0
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += int(np.sum(vals == 0))
    return total


def _validate(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return False
    if not np.all(np.abs(points - np.round(points)) < 1e-6):
        return False
    if not np.all(np.abs(lines - np.round(lines)) < 1e-6):
        return False
    pts_set = set(map(tuple, points.astype(np.int64).tolist()))
    if len(pts_set) != 54:
        return False
    for l in lines:
        if l[0] == 0 and l[1] == 0:
            return False
    norm_lines = set()
    for l in lines:
        nl = _normalize_line(l[0], l[1], l[2])
        if nl in norm_lines:
            return False
        norm_lines.add(nl)
    return True


def _line_through_two_points(p1, p2):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x1 == x2 and y1 == y2:
        return None
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _line_incidents_vec(points, lines):
    """Return per-line incidence counts as a vector."""
    counts = np.zeros(len(lines), dtype=np.int64)
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        counts[i] = int(np.sum(vals == 0))
    return counts


def _build_grid_and_lines(gx, gy, n_diag=12):
    """Build a gx x gy grid with all vertical/horizontal lines and best diagonal lines.
    Uses BOTH slope 1 (y = x + c) and slope -1 (y = -x + c) diagonals for diversity."""
    xs = np.arange(1, gx + 1)
    ys = np.arange(1, gy + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    for i in range(1, gx + 1):
        lines.append((1, 0, -i))
    for j in range(1, gy + 1):
        lines.append((0, 1, -j))

    # Slope 1 lines: y = x + c -> (1, -1, c), c = y - x
    slope1 = []
    for c in range(-(gy - 1), gx):
        count = 0
        for x in range(1, gx + 1):
            y = x + c
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            slope1.append((count, c))
    slope1.sort(reverse=True)

    # Slope -1 lines: y = -x + c -> (1, 1, -c), c = x + y
    slope_neg1 = []
    for c in range(2, gx + gy + 1):
        count = 0
        for x in range(1, gx + 1):
            y = c - x
            if 1 <= y <= gy:
                count += 1
        if count > 0:
            slope_neg1.append((count, c))
    slope_neg1.sort(reverse=True)

    # Interleave: take best from each family alternately to maximize diversity
    s1_idx = 0
    sn1_idx = 0
    diag_lines = []
    while len(diag_lines) < n_diag:
        if s1_idx < len(slope1) and (sn1_idx >= len(slope_neg1) or
                (s1_idx < len(slope1) and slope1[s1_idx][0] >= slope_neg1[sn1_idx][0])):
            c = slope1[s1_idx][1]
            diag_lines.append((1, -1, c))
            s1_idx += 1
        elif sn1_idx < len(slope_neg1):
            c = slope_neg1[sn1_idx][1]
            diag_lines.append((1, 1, -c))
            sn1_idx += 1
        elif s1_idx < len(slope1):
            c = slope1[s1_idx][1]
            diag_lines.append((1, -1, c))
            s1_idx += 1
        else:
            break

    lines.extend(diag_lines)

    # Pad to 27 if needed (shouldn't happen for valid gx+gy+n_diag=27)
    while len(lines) < 27:
        added = False
        for count, c in slope1:
            if (1, -1, c) not in lines:
                lines.append((1, -1, c))
                added = True
                break
        if not added:
            for count, c in slope_neg1:
                if (1, 1, -c) not in lines:
                    lines.append((1, 1, -c))
                    added = True
                    break
        if not added:
            lines.append((1, 2, 1))

    lines = lines[:27]
    return points, np.array(lines, dtype=np.int64)


def _reoptimize_points(points, lines, deadline, max_passes=2):
    """Greedy: for each point, find a nearby unoccupied position that increases its line count."""
    current_points = points.copy()
    current_score = _count_incidents(current_points, lines)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        if time.time() > deadline:
            break
        for idx in range(len(current_points)):
            if time.time() > deadline:
                break
            orig_x, orig_y = int(current_points[idx, 0]), int(current_points[idx, 1])
            old_on = 0
            for l in lines:
                A, B, C = l
                if A * orig_x + B * orig_y + C == 0:
                    old_on += 1
            best_new_pos = None
            best_new_on = old_on
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig_x + dx, orig_y + dy
                    occupied = False
                    for other in range(len(current_points)):
                        if other != idx and (int(current_points[other, 0]), int(current_points[other, 1])) == (nx, ny):
                            occupied = True
                            break
                    if occupied:
                        continue
                    new_on = 0
                    for l in lines:
                        A, B, C = l
                        if A * nx + B * ny + C == 0:
                            new_on += 1
                    if new_on > best_new_on:
                        best_new_on = new_on
                        best_new_pos = (nx, ny)
            if best_new_pos is not None:
                current_points[idx] = best_new_pos
                current_score += (best_new_on - old_on)
                improved = True
    if current_score > _count_incidents(points, lines):
        return current_points
    return None


def _generate_candidate_lines(points, exclude_set, max_candidates=80):
    candidates = []
    seen = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            nl = _line_through_two_points(points[i], points[j])
            if nl is None:
                continue
            if nl in exclude_set or nl in seen:
                continue
            seen.add(nl)
            candidates.append(nl)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _local_search(points, lines, deadline, max_iters=200):
    """Local search with line swaps + point re-optimization."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _count_incidents(best_points, best_lines)

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        line_incidents = _line_incidents_vec(best_points, best_lines)
        min_line_idx = int(np.argmin(line_incidents))
        min_inc = int(line_incidents[min_line_idx])

        exclude_set = set()
        for i in range(len(best_lines)):
            if i != min_line_idx:
                exclude_set.add(_normalize_line(*best_lines[i]))

        candidates = _generate_candidate_lines(best_points, exclude_set, max_candidates=60)
        best_candidate = None
        best_candidate_score = best_score

        for cand in candidates:
            A, B, C = cand
            vals = A * best_points[:, 0] + B * best_points[:, 1] + C
            cand_inc = int(np.sum(vals == 0))
            if cand_inc <= min_inc:
                continue
            new_lines = best_lines.copy()
            new_lines[min_line_idx] = np.array(cand, dtype=np.int64)
            new_points = _reoptimize_points(best_points, new_lines, deadline)
            if new_points is None:
                continue
            score = _count_incidents(new_points, new_lines)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = (new_points.copy(), new_lines.copy())

        if best_candidate is not None:
            best_points, best_lines = best_candidate
            best_score = best_candidate_score
        else:
            new_points = _reoptimize_points(best_points, best_lines, deadline)
            if new_points is not None:
                score = _count_incidents(new_points, best_lines)
                if score > best_score:
                    best_points = new_points
                    best_score = score
            else:
                break

    return best_points, best_lines, best_score


def _simulated_annealing(points, lines, deadline, start_temp=5.0):
    """Simulated annealing with 2-point line moves: pick two points, move them onto
    a line through two other points. This is a structural move the current hill-climber
    cannot make (single-point moves are too local)."""
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidents(current_points, current_lines)
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score

    n = len(current_points)
    start_time = time.time()
    total_time = max(1.0, deadline - start_time)
    end_time = start_time + total_time
    # Linear cooling
    temp = start_temp
    final_temp = 0.1
    rng = np.random.default_rng(42)

    while time.time() < end_time:
        elapsed_frac = (time.time() - start_time) / total_time
        temp = start_temp + (final_temp - start_temp) * elapsed_frac

        # Pick two points to move
        i, j = rng.choice(n, size=2, replace=False)
        # Pick two anchor points (distinct from i, j)
        anchors = [k for k in range(n) if k != i and k != j]
        a, b = rng.choice(anchors, size=2, replace=False)

        line_ab = _line_through_two_points(current_points[a], current_points[b])
        if line_ab is None:
            continue
        A, B, C = line_ab

        # Find two distinct integer points on this line that are not occupied
        # and not equal to the original positions of i and j
        # Parametrize: if A != 0, x free, y = (-A*x - C)/B if B != 0, else x = -C/A
        # Easier: generate candidate positions along the line near the anchors
        occupied = set(map(tuple, current_points.astype(np.int64).tolist()))
        candidates = []
        # Try positions near the two anchors, on the line
        for anchor in [a, b]:
            ax, ay = int(current_points[anchor, 0]), int(current_points[anchor, 1])
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ax + dx, ay + dy
                    if A * nx + B * ny + C != 0:
                        continue
                    if (nx, ny) in occupied:
                        continue
                    if (nx, ny) == (int(current_points[i, 0]), int(current_points[i, 1])):
                        continue
                    if (nx, ny) == (int(current_points[j, 0]), int(current_points[j, 1])):
                        continue
                    candidates.append((nx, ny))
        if len(candidates) < 2:
            continue
        # Pick two distinct candidates
        c1, c2 = rng.choice(len(candidates), size=2, replace=False)
        pos_i = candidates[c1]
        pos_j = candidates[c2]
        if pos_i == pos_j:
            continue

        # Compute delta: change in line count for points i and j
        old_i_on = sum(1 for l in current_lines if l[0] * int(current_points[i, 0]) + l[1] * int(current_points[i, 1]) + l[2] == 0)
        old_j_on = sum(1 for l in current_lines if l[0] * int(current_points[j, 0]) + l[1] * int(current_points[j, 1]) + l[2] == 0)
        new_i_on = sum(1 for l in current_lines if l[0] * pos_i[0] + l[1] * pos_i[1] + l[2] == 0)
        new_j_on = sum(1 for l in current_lines if l[0] * pos_j[0] + l[1] * pos_j[1] + l[2] == 0)
        delta = (new_i_on - old_i_on) + (new_j_on - old_j_on)

        # Accept
        if delta >= 0 or rng.random() < np.exp(delta / max(temp, 1e-6)):
            current_points[i] = pos_i
            current_points[j] = pos_j
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                best_lines = current_lines.copy()

    return best_points, best_lines, best_score


def _try_configuration(points, lines, deadline):
    if not _validate(points, lines):
        return None, None, 0
    score = _count_incidents(points, lines)
    # Local search first
    new_points, new_lines, new_score = _local_search(points, lines, deadline)
    if new_score > score:
        points, lines, score = new_points, new_lines, new_score
    # Then simulated annealing to escape local minima
    if time.time() < deadline - 30:
        sa_points, sa_lines, sa_score = _simulated_annealing(points, lines, deadline)
        if sa_score > score:
            points, lines, score = sa_points, sa_lines, sa_score
        # One more local search pass after SA
        if time.time() < deadline - 10:
            new_points, new_lines, new_score = _local_search(points, lines, deadline)
            if new_score > score:
                points, lines, score = new_points, new_lines, new_score
    return points, lines, score


def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit

    # Multiple grid configurations with both slope-1 and slope-(-1) diagonals
    configs = []
    for gx, gy in [(6, 9), (9, 6), (3, 18), (18, 3), (5, 11), (11, 5), (4, 14), (14, 4)]:
        if gx + gy != 15:
            continue
        n_diag = 27 - (gx + gy)
        pts, lns = _build_grid_and_lines(gx, gy, n_diag)
        configs.append((pts, lns))

    # Also the hand-built inspiration config (6x9 with specific c values)
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pts_hb = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines_hb = []
    for i in range(1, 7):
        lines_hb.append((1, 0, -i))
    for j in range(1, 10):
        lines_hb.append((0, 1, -j))
    for c in [0, 1, 2, 3, -1, 4]:
        lines_hb.append((1, -1, c))
    for c in [7, 8, 9, 10, 6, 11]:
        lines_hb.append((1, 1, -c))
    configs.append((pts_hb, np.array(lines_hb, dtype=np.int64)))

    best_points, best_lines, best_score = None, None, 0
    for pts_c, lines_c in configs:
        if time.time() > deadline - 30:
            break
        p, l, s = _try_configuration(pts_c, lines_c, deadline)
        if p is not None and s > best_score:
            best_points, best_lines, best_score = p, l, s

    if best_points is None:
        # Fallback
        pts, lns = _build_grid_and_lines(6, 9, 12)
        return pts, lns

    # Final validation
    if not _validate(best_points, best_lines):
        pts, lns = _build_grid_and_lines(6, 9, 12)
        return pts, lns

    return best_points, best_lines
