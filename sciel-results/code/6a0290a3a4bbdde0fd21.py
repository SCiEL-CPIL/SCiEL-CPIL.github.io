# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A = -A; B = -B; C = -C
    elif A == 0 and B < 0:
        B = -B; C = -C
    return (A, B, C)

def _build_pool():
    """Build a large pool of candidate lines with small integer coefficients."""
    pool = set()
    # For each slope family (B, A) meaning line A*x + B*y + C = 0,
    # i.e. y = -(A/B) x - C/B. Add many intercepts C.
    # Represent slope by (A, B) with gcd(|A|,|B|)=1 and canonical sign.
    slopes = []
    # axis
    slopes.append((1, 0))   # x = -C
    slopes.append((0, 1))   # y = -C
    # diagonals and other rational slopes
    for B in range(1, 9):
        for A in range(-8, 9):
            if A == 0:
                continue
            g = gcd(abs(A), abs(B))
            if g != 1:
                continue
            a, b = A // g, B // g
            # canonical sign: make (a,b) first-nonzero positive
            if a < 0:
                a, b = -a, -b
            elif a == 0 and b < 0:
                b = -b
            slopes.append((a, b))
    # widen intercept ranges for shallow slopes, narrower for steep
    for (A, B) in slopes:
        # max |C| such that lines still cross in a reasonable box
        # intercept magnitude: for x in [-25,25], C ~ A*x + B*y
        if B == 0:
            cmax = 25
        elif A == 0:
            cmax = 25
        else:
            cmax = 30 + 3 * (abs(A) + abs(B))
        for C in range(-cmax, cmax + 1):
            pool.add(_normalize_line(A, B, C))
    pool.discard((0, 0, 0))
    return pool

def _intersections(lines):
    """All integer intersection points of pairs of lines, bounded box."""
    pts = set()
    n = len(lines)
    for i in range(n):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n):
            A2, B2, C2 = lines[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            if num_x % det != 0 or num_y % det != 0:
                continue
            x = num_x // det
            y = num_y // det
            if -40 <= x <= 40 and -40 <= y <= 40:
                pts.add((x, y))
    return pts

def _pick_points_optimal(lines, n_pts=54):
    """Pick the n_pts points that maximize total incidences with the given lines."""
    pts = _intersections(lines)
    if not pts:
        return []
    cand = list(pts)
    if len(cand) <= n_pts:
        return cand
    
    arr = np.array(cand, dtype=np.int64)
    px = arr[:, 0:1]
    py = arr[:, 1:2]
    lA = np.array([l[0] for l in lines], dtype=np.int64).reshape(1, -1)
    lB = np.array([l[1] for l in lines], dtype=np.int64).reshape(1, -1)
    lC = np.array([l[2] for l in lines], dtype=np.int64).reshape(1, -1)
    vals = px * lA + py * lB + lC
    inc = (vals == 0).astype(np.int64)
    deg = inc.sum(axis=1)
    
    # Greedy selection with consideration of line coverage balance
    selected = set()
    remaining = list(range(len(cand)))
    line_coverage = np.zeros(len(lines), dtype=np.int64)
    
    for _ in range(n_pts):
        if not remaining:
            break
        best_idx = -1
        best_score = -1
        for idx in remaining:
            point_deg = deg[idx]
            bonus = 0
            for li in range(len(lines)):
                if inc[idx, li] == 1 and line_coverage[li] < 3:
                    bonus += 1
            total = point_deg + bonus
            if total > best_score:
                best_score = total
                best_idx = idx
        if best_idx >= 0:
            selected.add(best_idx)
            line_coverage += inc[best_idx]
            remaining.remove(best_idx)
    
    return [cand[i] for i in selected]

def _score(points, lines):
    """Exact incidence count."""
    px = np.array([p[0] for p in points], dtype=np.int64).reshape(-1, 1)
    py = np.array([p[1] for p in points], dtype=np.int64).reshape(-1, 1)
    lA = np.array([l[0] for l in lines], dtype=np.int64).reshape(1, -1)
    lB = np.array([l[1] for l in lines], dtype=np.int64).reshape(1, -1)
    lC = np.array([l[2] for l in lines], dtype=np.int64).reshape(1, -1)
    vals = px * lA + py * lB + lC
    return int(np.sum(vals == 0))

def _initial_lines(pool):
    """Greedy initial 27-line selection: pick lines that create dense intersections."""
    # Start with a grid core for guaranteed structure
    init = set()
    for c in range(0, 8):
        init.add(_normalize_line(1, 0, -c))
        init.add(_normalize_line(0, 1, -c))
    # Add diagonals
    for c in range(-6, 7):
        init.add(_normalize_line(1, 1, -c))
        init.add(_normalize_line(1, -1, -c))
    # Trim to 27
    lst = list(init)
    if len(lst) > 27:
        random.shuffle(lst)
        lst = lst[:27]
    # If still short, fill from pool
    if len(lst) < 27:
        rest = [l for l in pool if l not in set(lst)]
        random.shuffle(rest)
        lst.extend(rest[:27 - len(lst)])
    return set(lst[:27])

def _local_search(lines, pool, deadline):
    """Targeted hill-climbing with multi-swap moves."""
    current = set(lines)
    cur_lines = list(current)
    cur_pts = _pick_points_optimal(cur_lines)
    if len(cur_pts) < 54:
        used = set(cur_pts)
        while len(cur_pts) < 54:
            p = (random.randint(-40, 40), random.randint(-40, 40))
            if p not in used:
                cur_pts.append(p); used.add(p)
        cur_pts = cur_pts[:54]
    cur_score = _score(cur_pts, cur_lines)

    pool_list = list(pool)
    best_score = cur_score
    best_lines = set(current)
    best_pts = list(cur_pts)
    no_improve = 0
    temperature = 1.0

    while time.time() < deadline:
        improved = False
        order = list(current)
        random.shuffle(order)
        
        for l_out in order:
            if time.time() >= deadline:
                break
            
            px = np.array([p[0] for p in cur_pts], dtype=np.int64).reshape(-1, 1)
            py = np.array([p[1] for p in cur_pts], dtype=np.int64).reshape(-1, 1)
            
            if len(pool_list) > 5000:
                sample = random.sample(pool_list, 5000)
            else:
                sample = pool_list
            
            cov = {}
            for l in sample:
                A, B, C = l
                vals = px * A + py * B + C
                cov[l] = int(np.sum(vals == 0))
            
            cands = [(cov.get(l, 0), l) for l in sample if l not in current]
            if not cands:
                continue
            cands.sort(key=lambda t: -t[0])
            
            for _, l_in in cands[:12]:
                new_lines = (current - {l_out}) | {l_in}
                nl = list(new_lines)
                np_ = _pick_points_optimal(nl)
                if len(np_) < 54:
                    used = set(np_)
                    while len(np_) < 54:
                        p = (random.randint(-40, 40), random.randint(-40, 40))
                        if p not in used:
                            np_.append(p); used.add(p)
                    np_ = np_[:54]
                s = _score(np_, nl)
                if s > cur_score:
                    current = new_lines
                    cur_lines = nl
                    cur_pts = np_
                    cur_score = s
                    if s > best_score:
                        best_score = s
                        best_lines = set(new_lines)
                        best_pts = list(np_)
                    improved = True
                    no_improve = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve >= 2:
                if len(current) >= 2:
                    to_remove = random.sample(list(current), 2)
                    available = [l for l in pool_list if l not in current]
                    if available:
                        to_add = random.sample(available, 2)
                        new_lines = (current - set(to_remove)) | set(to_add)
                        current = new_lines
                        cur_lines = list(new_lines)
                        cur_pts = _pick_points_optimal(cur_lines)
                        if len(cur_pts) < 54:
                            used = set(cur_pts)
                            while len(cur_pts) < 54:
                                p = (random.randint(-40, 40), random.randint(-40, 40))
                                if p not in used:
                                    cur_pts.append(p); used.add(p)
                            cur_pts = cur_pts[:54]
                        cur_score = _score(cur_pts, cur_lines)
                no_improve = 0

    return best_lines, best_pts, best_score

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 590  # safety margin before 630s kill

    pool = _build_pool()

    # Multiple random restarts to escape local optima
    best_lines = None
    best_pts = None
    best_score = -1

    n_restarts = 8
    per_restart = (deadline - start) / max(1, n_restarts)

    for r in range(n_restarts):
        t0 = time.time()
        init = _initial_lines(pool)
        dl = min(deadline - 5, t0 + per_restart)
        bl, bp, bs = _local_search(init, pool, dl)
        if bs > best_score:
            best_score = bs
            best_lines = set(bl)
            best_pts = list(bp)
        if time.time() >= deadline - 5:
            break

    # Final polish: re-pick points for the best line set
    final_lines = list(best_lines)
    final_pts = _pick_points_optimal(final_lines)
    if len(final_pts) < 54:
        used = set(final_pts)
        while len(final_pts) < 54:
            p = (random.randint(-35, 35), random.randint(-35, 35))
            if p not in used:
                final_pts.append(p); used.add(p)
        final_pts = final_pts[:54]
    else:
        final_pts = final_pts[:54]

    # One more local search on the polished points if time remains
    if time.time() < deadline - 10:
        bl, bp, bs = _local_search(set(final_lines), pool, deadline - 5)
        if bs > best_score:
            best_score = bs
            best_lines = set(bl)
            best_pts = list(bp)
            final_lines = list(best_lines)
            final_pts = _pick_points_optimal(final_lines)
            if len(final_pts) < 54:
                used = set(final_pts)
                while len(final_pts) < 54:
                    p = (random.randint(-35, 35), random.randint(-35, 35))
                    if p not in used:
                        final_pts.append(p); used.add(p)
                final_pts = final_pts[:54]
            else:
                final_pts = final_pts[:54]

    # Build output arrays
    points_arr = np.array(final_pts, dtype=np.int64)
    lines_arr = np.array(list(best_lines), dtype=np.int64)

    # Validation / fallback
    if points_arr.shape != (54, 2) or lines_arr.shape != (27, 3):
        pts = [(i, j) for i in range(8) for j in range(7)]
        points_arr = np.array(pts, dtype=np.int64)
        ls = [(1, 0, -i) for i in range(8)] + [(0, 1, -j) for j in range(7)]
        lines_arr = np.array(ls, dtype=np.int64)

    return points_arr, lines_arr
# EVOLVE-BLOCK-END
