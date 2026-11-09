import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    def elapsed():
        return time.time() - t0

    def compute_max_area_count(pts):
        """Compute the maximum multiplicity of any non-zero doubled area."""
        n = len(pts)
        if n < 3:
            return 0

        P = np.array(pts, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(n, k=1)
        d = P[j_idx] - P[i_idx]
        dx = d[:, 0]
        dy = d[:, 1]
        M = len(dx)

        area_counts = defaultdict(int)
        for u in range(M - 1):
            cu = dx[u]
            cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if len(nz) == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                area_counts[int(v)] += int(c)

        if not area_counts:
            return 0
        return max(area_counts.values())

    def score_of(pts):
        return compute_max_area_count(pts)

    # ---------- Candidate generation ----------
    candidates = []

    def add_if_valid(cand):
        if len(cand) != 30:
            return
        key = tuple(sorted(cand))
        # quick duplicate check via set of candidates
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append(list(cand))

    seen_keys = set()

    # Two parallel horizontal lines (10+10+10 or 15+15)
    # 15+15
    for d in range(1, 15):
        for x1 in range(-30, 16):
            for x2 in range(-30, 16):
                if x1 + 15 > 30 or x2 + 15 > 30:
                    continue
                cand = [(x, 0) for x in range(x1, x1 + 15)] + \
                       [(x, d) for x in range(x2, x2 + 15)]
                add_if_valid(cand)

    # 10+10+10 three lines
    for d1 in range(1, 12):
        for d2 in range(d1 + 1, 24):
            for x1 in range(-30, 21):
                for x2 in range(-30, 21):
                    for x3 in range(-30, 21):
                        if x1 + 10 > 30 or x2 + 10 > 30 or x3 + 10 > 30:
                            continue
                        cand = [(x, 0) for x in range(x1, x1 + 10)] + \
                               [(x, d1) for x in range(x2, x2 + 10)] + \
                               [(x, d2) for x in range(x3, x3 + 10)]
                        add_if_valid(cand)

    # 4 lines of 8 (24) + 6 extra on existing lines
    for d1 in range(1, 8):
        for d2 in range(d1 + 1, 14):
            for d3 in range(d2 + 1, 20):
                for x1 in range(-30, 23):
                    for x2 in range(-30, 23):
                        for x3 in range(-30, 23):
                            for x4 in range(-30, 23):
                                if x1 + 8 > 30 or x2 + 8 > 30 or x3 + 8 > 30 or x4 + 8 > 30:
                                    continue
                                cand = [(x, 0) for x in range(x1, x1 + 8)] + \
                                       [(x, d1) for x in range(x2, x2 + 8)] + \
                                       [(x, d2) for x in range(x3, x3 + 8)] + \
                                       [(x, d3) for x in range(x4, x4 + 8)]
                                # add 6 more points on these lines (extend)
                                existing = set(cand)
                                lines = [0, d1, d2, d3]
                                added = 0
                                for ly in lines:
                                    for x in range(-30, 31):
                                        if added >= 6:
                                            break
                                        if (x, ly) not in existing:
                                            cand.append((x, ly))
                                            existing.add((x, ly))
                                            added += 1
                                    if added >= 6:
                                        break
                                if len(cand) == 30:
                                    add_if_valid(cand)

    # Random samples for diversity
    random.seed(42)
    for seed in range(15):
        random.seed(seed)
        cand = []
        used = set()
        while len(cand) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                cand.append((x, y))
        add_if_valid(cand)

    # ---------- Evaluate candidates ----------
    best_points = None
    best_score = 0

    for cand in candidates:
        if elapsed() > DEADLINE - 20.0:
            break
        s = score_of(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = score_of(best_points)

    # ---------- Local search ----------
    current_points = list(best_points)
    current_score = best_score

    random.seed(123)
    step_size = 3
    no_improve_count = 0
    iter_count = 0

    while elapsed() < DEADLINE:
        iter_count += 1
        r = random.random()

        if r < 0.30:
            # 2-point swap
            i = random.randint(0, 29)
            j = random.randint(0, 29)
            if i == j:
                continue
            xi, yi = current_points[i]
            xj, yj = current_points[j]
            new_i = (xj, yj)
            new_j = (xi, yi)
            # check collisions with others
            valid = True
            for k in range(30):
                if k == i or k == j:
                    continue
                if current_points[k] == new_i or current_points[k] == new_j:
                    valid = False
                    break
            if not valid:
                continue
            current_points[i] = new_i
            current_points[j] = new_j
            new_score = score_of(current_points)
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(current_points)
            else:
                current_points[i] = (xi, yi)
                current_points[j] = (xj, yj)
                no_improve_count += 1
        else:
            # single-point move
            idx = random.randint(0, 29)
            old_x, old_y = current_points[idx]

            r2 = random.random()
            if r2 < 0.03:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif r2 < 0.15:
                new_x = old_x + random.randint(-6, 6)
                new_y = old_y + random.randint(-6, 6)
            else:
                new_x = old_x + random.randint(-step_size, step_size)
                new_y = old_y + random.randint(-step_size, step_size)

            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                continue
            new_pt = (new_x, new_y)
            if new_pt == (old_x, old_y):
                continue

            is_valid = True
            for j in range(30):
                if j != idx and current_points[j] == new_pt:
                    is_valid = False
                    break
            if not is_valid:
                continue

            current_points[idx] = new_pt
            new_score = score_of(current_points)
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_points = list(current_points)
                    if step_size > 1 and random.random() < 0.2:
                        step_size = max(1, step_size - 1)
            else:
                current_points[idx] = (old_x, old_y)
                no_improve_count += 1

        if no_improve_count > 1500:
            current_points = list(best_points)
            current_score = best_score
            no_improve_count = 0
            step_size = random.randint(4, 8)

        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    return best_points
