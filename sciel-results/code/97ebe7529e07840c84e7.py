import time
import numpy as np
from collections import defaultdict
import random

def construct():
    t0 = time.time()
    DEADLINE = 58.0  # Safety margin before 60s limit

    def elapsed():
        return time.time() - t0

    N = 30
    _ii, _jj = np.triu_indices(N, k=1)
    _M = len(_ii)

    def area_counts_vec(pts):
        """Return (max multiplicity, dict area->count) for non-zero doubled areas."""
        P = np.asarray(pts, dtype=np.int64)
        d = P[_jj] - P[_ii]
        dx = d[:, 0]
        dy = d[:, 1]
        counts = defaultdict(int)
        for u in range(_M - 1):
            cu = dx[u]; cv = dy[u]
            if cu == 0 and cv == 0:
                continue
            cross = cu * dy[u+1:] - cv * dx[u+1:]
            nz = cross[cross != 0]
            if nz.size == 0:
                continue
            nz_abs = np.abs(nz)
            vals, cnts = np.unique(nz_abs, return_counts=True)
            for v, c in zip(vals, cnts):
                counts[int(v)] += int(c)
        if not counts:
            return 0, counts
        return max(counts.values()), counts

    def score_of(pts):
        return area_counts_vec(pts)[0]

    def top_areas(counts, k=5):
        if not counts:
            return set()
        return set(a for a, c in sorted(counts.items(), key=lambda x: -x[1])[:k] if c > 0)

    # ---- Candidate generation ----------------------------------------------
    candidates = []
    seen = set()

    def add(cand):
        if len(cand) != N:
            return
        if len(set(cand)) != N:
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        candidates.append(cand)

    # Lattice grids (varied shapes)
    for w in range(3, 11):
        for h in range(3, 11):
            total = w * h
            if total < N - 6 or total > N + 6:
                continue
            for x0 in range(-30, 31 - w):
                for y0 in range(-30, 31 - h):
                    cand = [(x, y) for x in range(x0, x0 + w) for y in range(y0, y0 + h)]
                    if len(cand) == N:
                        add(cand)
                    elif len(cand) < N:
                        existing = set(cand)
                        need = N - len(cand)
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                for x in range(x0, x0 + w):
                                    for y in range(y0, y0 + h):
                                        if need <= 0:
                                            break
                                        nx, ny = x + dx, y + dy
                                        if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in existing:
                                            cand.append((nx, ny)); existing.add((nx, ny)); need -= 1
                                        if need <= 0:
                                            break
                                    if need <= 0:
                                        break
                                if need <= 0:
                                    break
                            if need <= 0:
                                break
                        if len(cand) == N:
                            add(cand)

    # Parallel-line families (2, 3, 4 lines)
    for nlines in (2, 3, 4):
        for per in range(6, 16):
            if nlines * per < N - 4 or nlines * per > N + 4:
                continue
            for gap in (1, 2, 3, 4):
                for x0 in range(-30, 31 - per):
                    base = []
                    for L in range(nlines):
                        base += [(x, L * gap) for x in range(x0, x0 + per)]
                    if len(base) == N:
                        add(base)
                    elif len(base) < N:
                        existing = set(base)
                        need = N - len(base)
                        for L in range(nlines):
                            for x in range(x0 - 2, x0 + per + 2):
                                if need <= 0:
                                    break
                                if -30 <= x <= 30 and (x, L * gap) not in existing:
                                    base.append((x, L * gap)); existing.add((x, L * gap)); need -= 1
                        if len(base) == N:
                            add(base)

    # Diagonal / sheared lattices
    for a in (1, 2, 3):
        for b in (1, 2, 3):
            for w in range(4, 9):
                for h in range(4, 9):
                    if w * h < N - 4 or w * h > N + 4:
                        continue
                    for x0 in range(-30, 31):
                        for y0 in range(-30, 31):
                            cand = []
                            for i in range(w):
                                for j in range(h):
                                    x = x0 + i * a
                                    y = y0 + j * b
                                    if -30 <= x <= 30 and -30 <= y <= 30:
                                        cand.append((x, y))
                            if len(cand) == N:
                                add(cand)
                            elif len(cand) > N:
                                add(cand[:N])
                            elif len(cand) < N:
                                existing = set(cand)
                                need = N - len(cand)
                                for i in range(w):
                                    for j in range(h):
                                        if need <= 0:
                                            break
                                        for dx in range(-2, 3):
                                            for dy in range(-2, 3):
                                                if need <= 0:
                                                    break
                                                x = x0 + i * a + dx
                                                y = y0 + j * b + dy
                                                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in existing:
                                                    cand.append((x, y)); existing.add((x, y)); need -= 1
                                        if need <= 0:
                                            break
                                    if need <= 0:
                                        break
                                if len(cand) == N:
                                    add(cand)

    # Random samples
    for seed in range(15):
        random.seed(seed)
        cand = []
        used = set()
        while len(cand) < N:
            x = random.randint(-30, 30); y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y)); cand.append((x, y))
        add(cand)

    # ---- Evaluate candidates ------------------------------------------------
    best_points = None
    best_score = 0
    best_target = set()

    for cand in candidates:
        if elapsed() > DEADLINE:
            break
        s, counts = area_counts_vec(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
            best_target = top_areas(counts, 5)

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score, counts = area_counts_vec(best_points)
        best_target = top_areas(counts, 5)

    # ---- Local search refinement -------------------------------------------
    current_points = list(best_points)
    current_score = best_score
    current_target = set(best_target)

    random.seed(123)
    step_size = 3
    no_improve = 0
    iter_count = 0

    while elapsed() < DEADLINE:
        iter_count += 1
        idx = random.randint(0, N - 1)
        old = current_points[idx]
        ox, oy = old

        r = random.random()
        if r < 0.04:
            nx = random.randint(-30, 30); ny = random.randint(-30, 30)
        elif r < 0.18:
            nx = ox + random.randint(-6, 6); ny = oy + random.randint(-6, 6)
        else:
            nx = ox + random.randint(-step_size, step_size)
            ny = oy + random.randint(-step_size, step_size)

        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
            continue
        new_pt = (nx, ny)
        if new_pt == old:
            continue

        # collision check
        valid = True
        for j in range(N):
            if j != idx and current_points[j] == new_pt:
                valid = False
                break
        if not valid:
            continue

        current_points[idx] = new_pt
        new_score, new_counts = area_counts_vec(current_points)

        if new_score > current_score:
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best_score = new_score
                best_points = list(current_points)
                current_target = top_areas(new_counts, 5)
                if step_size > 1 and random.random() < 0.2:
                    step_size = max(1, step_size - 1)
        elif new_score == current_score and new_score > 0:
            # tie: accept if it increases target-area support (secondary signal)
            tc = sum(new_counts.get(a, 0) for a in current_target)
            if tc > 0:
                no_improve += 1
            else:
                current_points[idx] = old
                no_improve += 1
        else:
            current_points[idx] = old
            no_improve += 1

        if no_improve > 1200:
            current_points = list(best_points)
            current_score = best_score
            no_improve = 0
            step_size = random.randint(4, 8)

        if random.random() < 0.005:
            step_size = random.randint(1, 6)

    return best_points
