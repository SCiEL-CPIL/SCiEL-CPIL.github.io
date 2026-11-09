# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    return int(A), int(B), int(C)

def _build_incidence(points, lines):
    return (lines[:, 0][np.newaxis, :] * points[:, 0][:, np.newaxis] +
            lines[:, 1][np.newaxis, :] * points[:, 1][:, np.newaxis] +
            lines[:, 2][np.newaxis, :] == 0)

def _make_initial():
    k, m = 6, 9
    xs = np.arange(k)
    ys = np.arange(m)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    lines_list = []
    for i in range(k):
        lines_list.append((1, 0, -i))
    for j in range(m):
        lines_list.append((0, 1, -j))
    for c in range(-8, 6):
        if len(lines_list) >= 27:
            break
        lines_list.append((1, -1, -c))
    lines_list = lines_list[:27]
    lines = np.array(lines_list, dtype=np.int64)
    return points, lines

def _dedupe_points(points):
    seen = set()
    out = []
    for p in points:
        t = (int(p[0]), int(p[1]))
        if t not in seen:
            seen.add(t)
            out.append(t)
    while len(out) < 54:
        x = random.randint(-3, 10)
        y = random.randint(-3, 12)
        if (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    return np.array(out[:54], dtype=np.int64)

def _dedupe_lines(lines, points, n_lines=27):
    seen = set()
    out = []
    for L in lines:
        A, B, C = int(L[0]), int(L[1]), int(L[2])
        if A == 0 and B == 0:
            continue
        n = _normalize_line(A, B, C)
        if n not in seen:
            seen.add(n)
            out.append([A, B, C])
    while len(out) < n_lines:
        a = random.randint(0, 53)
        b = random.randint(0, 53)
        while b == a:
            b = random.randint(0, 53)
        A, B, C = _line_from_points(points[a], points[b])
        if A == 0 and B == 0:
            continue
        n = _normalize_line(A, B, C)
        if n not in seen:
            seen.add(n)
            out.append([A, B, C])
    return np.array(out[:n_lines], dtype=np.int64)

def _optimal_line_for_pair(points, a, b, norm_lines, skip_j):
    x_a, y_a = points[a]
    x_b, y_b = points[b]
    A = y_b - y_a
    B = x_a - x_b
    C = -(A * x_a + B * y_a)
    if A == 0 and B == 0:
        return None
    norm = _normalize_line(int(A), int(B), int(C))
    for l_idx in range(27):
        if l_idx == skip_j:
            continue
        if norm_lines[l_idx] == norm:
            return None
    return (int(A), int(B), int(C)), norm

def _optimize(points, lines, deadline):
    n_points, n_lines = 54, 27
    points = points.copy()
    lines = lines.copy()
    norm_lines = [_normalize_line(int(L[0]), int(L[1]), int(L[2])) for L in lines]
    M = _build_incidence(points, lines)
    I = int(np.sum(M))
    best_I = I
    best_points = points.copy()
    best_lines = lines.copy()

    T = 3.0
    T_min = 0.01
    cooling = 0.9999

    # Precompute all pair lines (54*53/2 = 1431)
    pair_lines = {}
    for a in range(n_points):
        for b in range(a+1, n_points):
            A, B, C = _line_from_points(points[a], points[b])
            if A == 0 or B == 0 or (A != 0 and B != 0):
                norm = _normalize_line(A, B, C)
                pair_lines[(a, b)] = (A, B, C, norm)

    # Build list of candidate lines for each point (all lines through that point)
    # For speed, we use random sampling

    while time.time() < deadline:
        # Point move
        if random.random() < 0.45:
            i = random.randint(0, n_points - 1)
            old_px, old_py = points[i, 0], points[i, 1]
            old_count = int(np.sum(M[i]))
            best_delta = -1000
            best_pos = None
            # Sample candidate positions: near current + on existing lines + random
            candidates = set()
            # Near current
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    candidates.add((old_px + dx, old_py + dy))
            # On existing lines (sample points on each line)
            for j in range(n_lines):
                A, B, C = lines[j, 0], lines[j, 1], lines[j, 2]
                # Find a point on this line near the grid
                # A*x + B*y = -C
                if A != 0:
                    for x_try in range(-2, 9):
                        num = -C - A * x_try
                        if B != 0 and num % B == 0:
                            y_try = num // B
                            if -3 <= y_try <= 11:
                                candidates.add((x_try, y_try))
                elif B != 0:
                    for y_try in range(-3, 12):
                        num = -C - B * y_try
                        if A != 0 and num % A == 0:
                            x_try = num // A
                            if -3 <= x_try <= 8:
                                candidates.add((x_try, y_try))
            # Random
            for _ in range(10):
                candidates.add((random.randint(-3, 10), random.randint(-3, 12)))

            for (cand_x, cand_y) in list(candidates)[:60]:
                if (cand_x, cand_y) == (old_px, old_py):
                    continue
                # Check duplicate
                dup = False
                for k in range(n_points):
                    if k == i:
                        continue
                    if points[k, 0] == cand_x and points[k, 1] == cand_y:
                        dup = True
                        break
                if dup:
                    continue
                new_M_i = (lines[:, 0] * cand_x + lines[:, 1] * cand_y + lines[:, 2] == 0)
                new_count = int(np.sum(new_M_i))
                delta = new_count - old_count
                if delta > best_delta:
                    best_delta = delta
                    best_pos = (cand_x, cand_y, new_M_i)

            if best_pos is None:
                continue

            cand_x, cand_y, new_M_i = best_pos
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                M[i] = new_M_i
                points[i, 0] = cand_x
                points[i, 1] = cand_y
                I += best_delta
                # Update pair_lines for pairs involving i
                for k in range(n_points):
                    if k == i:
                        continue
                    A, B, C = _line_from_points(points[i], points[k])
                    if A == 0 and B == 0:
                        continue
                    norm = _normalize_line(A, B, C)
                    key = (min(i, k), max(i, k))
                    pair_lines[key] = (A, B, C, norm)
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()
        else:
            # Line retargeting: pick a pair of points and retarget a line
            j = random.randint(0, n_lines - 1)
            old_count = int(np.sum(M[:, j]))
            best_delta = -1000
            best_line = None
            best_norm = None
            # Sample random pairs
            for _ in range(40):
                a = random.randint(0, n_points - 1)
                b = random.randint(0, n_points - 1)
                while b == a:
                    b = random.randint(0, n_points - 1)
                key = (min(a, b), max(a, b))
                if key not in pair_lines:
                    continue
                A, B, C, norm = pair_lines[key]
                # Check duplicate with other lines
                is_dup = False
                for l_idx in range(n_lines):
                    if l_idx == j:
                        continue
                    if norm_lines[l_idx] == norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
                new_M_j = (A * points[:, 0] + B * points[:, 1] + C == 0)
                new_count = int(np.sum(new_M_j))
                delta = new_count - old_count
                if delta > best_delta:
                    best_delta = delta
                    best_line = (A, B, C)
                    best_norm = norm

            if best_line is None:
                continue

            A_new, B_new, C_new = best_line
            new_M_j = (A_new * points[:, 0] + B_new * points[:, 1] + C_new == 0)
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                M[:, j] = new_M_j
                lines[j, 0] = A_new
                lines[j, 1] = B_new
                lines[j, 2] = C_new
                norm_lines[j] = best_norm
                I += best_delta
                if I > best_I:
                    best_I = I
                    best_points = points.copy()
                    best_lines = lines.copy()

        T *= cooling
        if T < T_min:
            T = T_min

    return best_points, best_lines, best_I

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 570.0

    best_points, best_lines, best_I = None, None, -1

    # Multiple restarts with different initial seeds
    random.seed(42)
    while time.time() < deadline:
        # Create initial configuration
        points, lines = _make_initial()
        # Add some randomness to initial points
        for _ in range(random.randint(5, 20)):
            i = random.randint(0, 53)
            points[i, 0] += random.randint(-1, 1)
            points[i, 1] += random.randint(-1, 1)
        points = _dedupe_points(points)
        lines = _dedupe_lines(lines, points)

        # Optimize this configuration
        time_left = deadline - time.time()
        if time_left < 30:
            break
        # Spend proportional time on this restart
        alloc = min(time_left - 5, 120.0)
        if alloc < 20:
            alloc = max(10, time_left - 2)
        local_deadline = time.time() + alloc
        pts, lns, I = _optimize(points, lines, local_deadline)

        if I > best_I:
            best_I = I
            best_points = pts
            best_lines = lns

    # Final validation and deduplication
    if best_points is None:
        best_points, best_lines = _make_initial()

    best_points = _dedupe_points(best_points)
    best_lines = _dedupe_lines(best_lines, best_points)

    # Final incidence check
    M = _build_incidence(best_points, best_lines)
    final_I = int(np.sum(M))

    return best_points, best_lines

# EVOLVE-BLOCK-END
