import time
import random
from collections import Counter
import numpy as np

_START = time.time()
_DEADLINE = 85.0  # safety margin under 90s kill

# Precompute indices for all C(30,3) triples
_TRIPLE_INDICES = None
_TRIPLE_COUNT = 0

def _get_triple_indices():
    global _TRIPLE_INDICES, _TRIPLE_COUNT
    if _TRIPLE_INDICES is None:
        i, j, k = np.indices((30, 30, 30))
        mask = (i < j) & (j < k)
        _TRIPLE_INDICES = (i[mask].astype(np.int64), j[mask].astype(np.int64), k[mask].astype(np.int64))
        _TRIPLE_COUNT = len(_TRIPLE_INDICES[0])
    return _TRIPLE_INDICES

def _score(points):
    """Return (max multiplicity, area) over positive doubled areas, or (0,0)."""
    n = len(points)
    if n < 3:
        return (0, 0)
    P = np.array(points, dtype=np.int64)
    I, J, K = _get_triple_indices()
    x1 = P[I, 0]; y1 = P[I, 1]
    x2 = P[J, 0]; y2 = P[J, 1]
    x3 = P[K, 0]; y3 = P[K, 1]
    det = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    nz = det[det > 0]
    if nz.size == 0:
        return (0, 0)
    counts = np.bincount(nz)
    best = int(counts.max())
    best_area = int(np.argmax(counts))
    return (best, best_area)

def _score_fast(P, I, J, K):
    """Fast score from precomputed numpy array. Returns (max_mult, area)."""
    x1 = P[I, 0]; y1 = P[I, 1]
    x2 = P[J, 0]; y2 = P[J, 1]
    x3 = P[K, 0]; y3 = P[K, 1]
    det = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    nz = det[det > 0]
    if nz.size == 0:
        return (0, 0)
    counts = np.bincount(nz)
    best = int(counts.max())
    best_area = int(np.argmax(counts))
    return (best, best_area)

def _valid(points):
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for (x, y) in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
    return True

def _grid_points(w, h, sx, sy, cx, cy, shear=0):
    """Generate a grid of w*h points with optional shear."""
    pts = []
    for i in range(w):
        for j in range(h):
            x = cx + i * sx + j * shear
            y = cy + j * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts

def _local_search_fast(pts, rng, time_budget, neighborhood=5):
    """Local search using fast scoring with precomputed triple indices."""
    I, J, K = _get_triple_indices()
    P = np.array(pts, dtype=np.int64)
    cur_score = _score_fast(P, I, J, K)
    best_P = P.copy()
    best_score = cur_score
    t0 = time.time()
    cur_set = set(pts)
    
    # Precompute which triples involve each point index
    point_triples = [[] for _ in range(30)]
    for t in range(_TRIPLE_COUNT):
        i, j, k = int(I[t]), int(J[t]), int(K[t])
        point_triples[i].append(t)
        point_triples[j].append(t)
        point_triples[k].append(t)
    
    # Precompute area contributions for current configuration
    x1 = P[I, 0]; y1 = P[I, 1]
    x2 = P[J, 0]; y2 = P[J, 1]
    x3 = P[K, 0]; y3 = P[K, 1]
    all_dets = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    max_rounds = 500
    round_num = 0
    improved = True
    while improved and time.time() - t0 < time_budget and round_num < max_rounds:
        improved = False
        round_num += 1
        for idx in range(30):
            if time.time() - t0 >= time_budget:
                break
            old_x, old_y = int(P[idx, 0]), int(P[idx, 1])
            old_pt = (old_x, old_y)
            
            # Try moves in neighborhood
            found = False
            for dx in range(-neighborhood, neighborhood + 1):
                if found or time.time() - t0 >= time_budget:
                    break
                for dy in range(-neighborhood, neighborhood + 1):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_pt = (new_x, new_y)
                    if new_pt in cur_set:
                        continue
                    
                    # Update P
                    P[idx, 0] = new_x
                    P[idx, 1] = new_y
                    
                    # Recompute only dets involving point idx
                    for t in point_triples[idx]:
                        i, j, k = int(I[t]), int(J[t]), int(K[t])
                        xi, yi = int(P[i, 0]), int(P[i, 1])
                        xj, yj = int(P[j, 0]), int(P[j, 1])
                        xk, yk = int(P[k, 0]), int(P[k, 1])
                        all_dets[t] = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    
                    # Score
                    nz = all_dets[all_dets > 0]
                    if nz.size == 0:
                        ns, na = (0, 0)
                    else:
                        counts = np.bincount(nz)
                        ns = int(counts.max())
                        na = int(np.argmax(counts))
                    
                    if (ns, na) > best_score:
                        best_score = (ns, na)
                        best_P = P.copy()
                        cur_set.discard(old_pt)
                        cur_set.add(new_pt)
                        improved = True
                        found = True
                        break
                    elif (ns, na) > cur_score:
                        cur_score = (ns, na)
                        cur_set.discard(old_pt)
                        cur_set.add(new_pt)
                        improved = True
                        found = True
                        break
                    else:
                        # Revert
                        P[idx, 0] = old_x
                        P[idx, 1] = old_y
                        for t in point_triples[idx]:
                            i, j, k = int(I[t]), int(J[t]), int(K[t])
                            xi, yi = int(P[i, 0]), int(P[i, 1])
                            xj, yj = int(P[j, 0]), int(P[j, 1])
                            xk, yk = int(P[k, 0]), int(P[k, 1])
                            all_dets[t] = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
            
            # Try swap moves: swap point idx with another point
            if not found and time.time() - t0 < time_budget:
                for jdx in range(idx + 1, 30):
                    if time.time() - t0 >= time_budget:
                        break
                    # Swap positions
                    P[idx, 0], P[jdx, 0] = P[jdx, 0], P[idx, 0]
                    P[idx, 1], P[jdx, 1] = P[jdx, 1], P[idx, 1]
                    
                    # Recompute dets involving idx or jdx
                    affected = set(point_triples[idx]) | set(point_triples[jdx])
                    for t in affected:
                        i, j, k = int(I[t]), int(J[t]), int(K[t])
                        xi, yi = int(P[i, 0]), int(P[i, 1])
                        xj, yj = int(P[j, 0]), int(P[j, 1])
                        xk, yk = int(P[k, 0]), int(P[k, 1])
                        all_dets[t] = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                    
                    nz = all_dets[all_dets > 0]
                    if nz.size == 0:
                        ns, na = (0, 0)
                    else:
                        counts = np.bincount(nz)
                        ns = int(counts.max())
                        na = int(np.argmax(counts))
                    
                    if (ns, na) > best_score:
                        best_score = (ns, na)
                        best_P = P.copy()
                        improved = True
                        found = True
                        break
                    elif (ns, na) > cur_score:
                        cur_score = (ns, na)
                        improved = True
                        found = True
                        break
                    else:
                        # Revert swap
                        P[idx, 0], P[jdx, 0] = P[jdx, 0], P[idx, 0]
                        P[idx, 1], P[jdx, 1] = P[jdx, 1], P[idx, 1]
                        for t in affected:
                            i, j, k = int(I[t]), int(J[t]), int(K[t])
                            xi, yi = int(P[i, 0]), int(P[i, 1])
                            xj, yj = int(P[j, 0]), int(P[j, 1])
                            xk, yk = int(P[k, 0]), int(P[k, 1])
                            all_dets[t] = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
    
    best_pts = [(int(best_P[i, 0]), int(best_P[i, 1])) for i in range(30)]
    return best_pts, best_score


def construct():
    global _START
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    best = None
    best_score = (0, 0)
    
    I, J, K = _get_triple_indices()
    
    def try_config(pts):
        nonlocal best, best_score
        if not _valid(pts):
            return False
        P = np.array(pts, dtype=np.int64)
        sc, sa = _score_fast(P, I, J, K)
        if (sc, sa) > best_score:
            best_score = (sc, sa)
            best = list(pts)
            return True
        return False
    
    t_start = time.time()
    
    # Phase 1: Systematic grid exploration
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    for w, h in factors_30:
        if time.time() - t_start > _DEADLINE - 35:
            break
        for sx in range(1, min(31 // w + 1, 11)):
            for sy in range(1, min(31 // h + 1, 11)):
                if time.time() - t_start > _DEADLINE - 35:
                    break
                width = (w - 1) * sx
                height = (h - 1) * sy
                if width > 60 or height > 60:
                    continue
                for cx in range(-width // 2, width // 2 + 1, max(1, width // 4)):
                    for cy in range(-height // 2, height // 2 + 1, max(1, height // 4)):
                        pts = _grid_points(w, h, sx, sy, cx, cy)
                        if len(pts) == 30:
                            try_config(pts)
        
        # Sheared grids
        if time.time() - t_start > _DEADLINE - 35:
            break
        for sx in range(1, 5):
            for sy in range(1, 5):
                for shear in range(-3, 4):
                    if time.time() - t_start > _DEADLINE - 35:
                        break
                    width = (w - 1) * sx
                    height = (h - 1) * sy
                    if width > 60 or height > 60:
                        continue
                    for cx in range(-width // 2, width // 2 + 1, max(1, width // 3)):
                        for cy in range(-height // 2, height // 2 + 1, max(1, height // 3)):
                            pts = _grid_points(w, h, sx, sy, cx, cy, shear)
                            if len(pts) == 30:
                                try_config(pts)
    
    # Phase 2: Line-based constructions
    if best is None:
        # Use a 5x6 grid centered at origin for better initial score
        best = [(x, y) for x in range(-2, 3) for y in range(-3, 3)]
        P = np.array(best, dtype=np.int64)
        best_score = _score_fast(P, I, J, K)
    
    if time.time() - t_start < _DEADLINE - 25:
        line_configs = []
        # 3 lines
        for y0 in range(-20, 21, 3):
            for dy in range(2, 20):
                ys = [y0, y0 + dy, y0 + 2 * dy]
                if max(ys) > 30 or min(ys) < -30:
                    continue
                for (a, b, c) in [(10, 10, 10), (11, 10, 9), (9, 10, 11), (12, 9, 9), (8, 11, 11)]:
                    line_configs.append((ys, [a, b, c]))
        # 4 lines
        for y0 in range(-15, 16, 3):
            for dy in range(2, 15):
                ys = [y0, y0 + dy, y0 + 2 * dy, y0 + 3 * dy]
                if max(ys) > 30 or min(ys) < -30:
                    continue
                for (a, b, c, d) in [(8, 8, 7, 7), (7, 8, 8, 7), (9, 7, 8, 6), (10, 8, 6, 6)]:
                    line_configs.append((ys, [a, b, c, d]))
        # 5 lines
        for y0 in range(-10, 11, 2):
            for dy in range(2, 10):
                ys = [y0 + i * dy for i in range(5)]
                if max(ys) > 30 or min(ys) < -30:
                    continue
                for (a, b, c, d, e) in [(6, 6, 6, 6, 6), (7, 6, 6, 6, 5), (5, 6, 6, 7, 6)]:
                    line_configs.append((ys, [a, b, c, d, e]))
        
        sampled = 0
        for (ys, counts) in line_configs:
            if time.time() - t_start > _DEADLINE - 25:
                break
            for trial in range(3):
                if time.time() - t_start > _DEADLINE - 25:
                    break
                lines = []
                for (y, cnt) in zip(ys, counts):
                    base = np.linspace(-25, 25, cnt)
                    jitter = np_rng.integers(-2, 3, size=cnt)
                    offs = np.sort(base + jitter).astype(int)
                    uniq = []
                    for v in offs:
                        if not uniq or v != uniq[-1]:
                            uniq.append(int(v))
                    used = set(uniq)
                    while len(uniq) < cnt:
                        v = int(np_rng.integers(-28, 29))
                        if v not in used:
                            uniq.append(v)
                            used.add(v)
                    lines.append((y, uniq[:cnt]))
                pts = []
                for (y, offs) in lines:
                    pts.extend([(x, y) for x in offs])
                if len(pts) == 30 and len(set(pts)) == 30:
                    sampled += 1
                    try_config(pts)
    
    # Phase 3: Fast local search with wider neighborhood
    if best is not None and time.time() - t_start < _DEADLINE - 8:
        budget = min(_DEADLINE - 8 - time.time(), 30.0)
        lp, lsc = _local_search_fast(best, np_rng, time_budget=budget, neighborhood=5)
        if lsc > best_score:
            best_score = lsc
            best = lp
    
    # Phase 4: Polish with smaller neighborhood
    if best is not None and time.time() - t_start < _DEADLINE - 4:
        budget = min(_DEADLINE - 4 - time.time(), 15.0)
        lp, lsc = _local_search_fast(best, np_rng, time_budget=budget, neighborhood=3)
        if lsc > best_score:
            best_score = lsc
            best = lp
    
    # Phase 4b: Another polish pass with neighborhood 2 for fine-tuning
    if best is not None and time.time() - t_start < _DEADLINE - 2:
        budget = min(_DEADLINE - 2 - time.time(), 8.0)
        if budget > 2:
            lp, lsc = _local_search_fast(best, np_rng, time_budget=budget, neighborhood=2)
            if lsc > best_score:
                best_score = lsc
                best = lp
    
    # Phase 5: Random restarts with perturbation
    if best is not None and time.time() - t_start < _DEADLINE - 3:
        for restart in range(50):
            if time.time() - t_start > _DEADLINE - 3:
                break
            new_pts = list(best)
            n_perturb = int(np_rng.integers(1, 6))
            for _ in range(n_perturb):
                idx = int(np_rng.integers(0, 30))
                dx = int(np_rng.integers(-5, 6))
                dy = int(np_rng.integers(-5, 6))
                new_x = new_pts[idx][0] + dx
                new_y = new_pts[idx][1] + dy
                if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                    new_pts[idx] = (new_x, new_y)
            if len(set(new_pts)) == 30:
                P = np.array(new_pts, dtype=np.int64)
                sc, sa = _score_fast(P, I, J, K)
                if (sc, sa) > best_score:
                    best_score = (sc, sa)
                    best = new_pts
                    if time.time() - t_start < _DEADLINE - 4:
                        lp, lsc = _local_search_fast(best, np_rng, time_budget=min(3.0, _DEADLINE - 4 - time.time()), neighborhood=3)
                        if lsc > best_score:
                            best_score = lsc
                            best = lp
    
    # Final validity check
    if best is None or not _valid(best):
        best = [(x, y) for x in range(6) for y in range(5)]
    
    return best

_construct_baseline = [(x, y) for x in range(6) for y in range(5)]
