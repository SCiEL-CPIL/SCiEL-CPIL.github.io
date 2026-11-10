"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import Counter

_DEADLINE = 85.0  # safety margin before 90s kill
_START_TIME = time.time()

_N = 30
# Precompute triple indices once (C(30,3) = 4060)
_i_idx, _j_idx, _k_idx = [], [], []
for _i in range(_N - 2):
    for _j in range(_i + 1, _N - 1):
        for _k in range(_j + 1, _N):
            _i_idx.append(_i)
            _j_idx.append(_j)
            _k_idx.append(_k)
_I_IDX = np.array(_i_idx, dtype=np.int32)
_J_IDX = np.array(_j_idx, dtype=np.int32)
_K_IDX = np.array(_k_idx, dtype=np.int32)
_NUM_TRIPLES = len(_I_IDX)


def _dets(pts):
    """All C(30,3) absolute doubled areas for a (30,2) int array."""
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    return np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))


def _score_fast(points):
    """Return (max_count, {area: count}) for positive areas."""
    pts = np.asarray(points, dtype=np.int64).reshape(30, 2)
    det = _dets(pts)
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
    vals, counts = np.unique(det, return_counts=True)
    area_dict = dict(zip(vals.tolist(), counts.tolist()))
    return int(counts.max()), area_dict


def _softmax(counts, beta=0.30):
    """Smooth surrogate for max(counts): temperature-soft max via log-sum-exp."""
    c = np.asarray(counts, dtype=np.float64)
    if c.size == 0:
        return 0.0
    m = c.max()
    if m <= 0:
        return 0.0
    return float(m + np.log(np.sum(np.exp((c - m) / beta))))


def _dets_involving(pts, idx):
    """Determinants for triples that include point idx. Returns array of C(n-1,2) values."""
    n = _N
    others = [i for i in range(n) if i != idx]
    m = len(others)  # n-1
    # For each pair (a,b) from others, compute det(pts[idx], pts[a], pts[b])
    # There are C(m,2) such pairs
    i_idx = []
    j_idx = []
    k_idx = []
    for ai in range(m):
        for bi in range(ai + 1, m):
            a = others[ai]
            b = others[bi]
            # Sort to get canonical triple
            triple = sorted((idx, a, b))
            i_idx.append(triple[0])
            j_idx.append(triple[1])
            k_idx.append(triple[2])
    if not i_idx:
        return np.zeros(0, dtype=np.int64)
    ia = np.array(i_idx, dtype=np.int32)
    ja = np.array(j_idx, dtype=np.int32)
    ka = np.array(k_idx, dtype=np.int32)
    xi = pts[ia, 0]; yi = pts[ia, 1]
    xj = pts[ja, 0]; yj = pts[ja, 1]
    xk = pts[ka, 0]; yk = pts[ka, 1]
    return np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))


def _gen_rect_grid(rows, cols, dx, dy):
    pts = set()
    total_w = (cols - 1) * dx
    total_h = (rows - 1) * dy
    if total_w > 60 or total_h > 60:
        return None
    ox = -total_w // 2
    oy = -total_h // 2
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    if len(pts) < 30:
        return None
    return sorted(pts)[:30]


def _gen_lattice(a, b, c, d, count=30):
    pts = set()
    for i in range(-12, 13):
        for j in range(-12, 13):
            x = i * a + j * c
            y = i * b + j * d
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    if len(pts) < count:
        return None
    return sorted(pts)[:count]


def _gen_hex_config(radius):
    pts = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if abs(s) > radius:
                continue
            x = q + s // 2
            y = r
            cx = x * 2
            cy = y * 2 + (q % 2)
            if -30 <= cx <= 30 and -30 <= cy <= 30:
                pts.add((cx, cy))
    if len(pts) < 30:
        return None
    return sorted(pts)[:30]


def _gen_initial_configs():
    configs = []
    for rows in range(3, 9):
        for cols in range(3, 11):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    cfg = _gen_rect_grid(rows, cols, dx, dy)
                    if cfg and len(cfg) == 30:
                        configs.append(cfg)
    for r in range(3, 6):
        cfg = _gen_hex_config(r)
        if cfg:
            configs.append(cfg)
    for a, b in [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (3, 1)]:
        for c, d in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (1, 3)]:
            if a * d - b * c == 0:
                continue
            cfg = _gen_lattice(a, b, c, d)
            if cfg:
                configs.append(cfg)
    rng = np.random.RandomState(42)
    for _ in range(12):
        pts = set()
        while len(pts) < 30:
            pts.add((int(rng.randint(-30, 31)), int(rng.randint(-30, 31))))
        configs.append(sorted(pts))
    unique = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def _local_search(points, deadline, target_area=None, beta=0.30):
    """SA local search using smooth surrogate objective + incremental scoring."""
    pts = np.asarray(points, dtype=np.int64).reshape(30, 2)
    det = _dets(pts)
    det = det[det > 0]
    vals, counts = np.unique(det, return_counts=True)
    area_counts = {int(v): int(c) for v, c in zip(vals, counts)}
    current_score = int(counts.max()) if counts.size else 0
    best_score = current_score
    best_points = pts.copy()
    best_counts = counts.copy()
    best_area_counts = dict(area_counts)
    if target_area is None and area_counts:
        target_area = max(area_counts, key=area_counts.get)

    n = _N
    point_set = set(map(tuple, pts.tolist()))
    temp = 5.0
    cooling = 0.996
    min_temp = 0.02
    rng = np.random.RandomState(7)

    def cur_soft():
        """Soft-max surrogate: rewards having one area with very high count."""
        c = np.array(list(area_counts.values()), dtype=np.float64)
        if c.size == 0:
            return 0.0
        m = c.max()
        if m <= 0:
            return 0.0
        # Use a sharper softmax to better approximate max
        return float(m + np.log(np.sum(np.exp((c - m) / beta))))

    cur_softv = cur_soft()
    best_softv = cur_softv

    while time.time() < deadline and temp > min_temp:
        idx = int(rng.randint(n))
        old_pt = tuple(pts[idx])
        u = rng.random()
        if u < 0.6:
            radius = int(rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
            new_x = old_pt[0] + int(rng.randint(-radius, radius + 1))
            new_y = old_pt[1] + int(rng.randint(-radius, radius + 1))
        elif u < 0.85:
            new_x = old_pt[0] + int(rng.randint(-10, 11))
            new_y = old_pt[1] + int(rng.randint(-10, 11))
        else:
            new_x = int(rng.randint(-30, 31))
            new_y = int(rng.randint(-30, 31))
        new_pt = (new_x, new_y)
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            temp *= cooling
            continue
        if new_pt != old_pt and new_pt in point_set:
            temp *= cooling
            continue

        # Incremental update of counts for the moved point
        old_det = _dets_involving(pts, idx)
        pts[idx] = new_pt
        new_det = _dets_involving(pts, idx)

        ov, oc = np.unique(old_det, return_counts=True)
        for v, c in zip(ov, oc):
            v = int(v)
            if v > 0:
                area_counts[v] = area_counts.get(v, 0) - int(c)
                if area_counts[v] <= 0:
                    area_counts.pop(v, None)
        nv, nc = np.unique(new_det, return_counts=True)
        for v, c in zip(nv, nc):
            v = int(v)
            if v > 0:
                area_counts[v] = area_counts.get(v, 0) + int(c)

        new_score = max(area_counts.values()) if area_counts else 0
        new_softv = cur_soft()

        if new_score > current_score or new_softv > cur_softv:
            current_score = new_score
            cur_softv = new_softv
            if new_score > best_score or new_softv > best_softv:
                best_score = new_score
                best_softv = new_softv
                best_points = pts.copy()
                best_counts = np.array(list(area_counts.values()), dtype=np.int64)
                best_area_counts = dict(area_counts)
        else:
            delta = new_softv - cur_softv
            prob = np.exp(delta / max(temp, 0.05))
            if rng.random() > prob:
                # revert
                for v, c in zip(nv, nc):
                    v = int(v)
                    if v > 0:
                        area_counts[v] = area_counts.get(v, 0) - int(c)
                        if area_counts[v] <= 0:
                            area_counts.pop(v, None)
                for v, c in zip(ov, oc):
                    v = int(v)
                    if v > 0:
                        area_counts[v] = area_counts.get(v, 0) + int(c)
                pts[idx] = old_pt
        point_set.discard(old_pt)
        point_set.add(new_pt)
        temp *= cooling

    return best_points.tolist(), best_score


def _try_all_configs():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    configs = _gen_initial_configs()

    best_score = 0
    best_points = configs[0] if configs else [(i, 0) for i in range(30)]
    target_areas = set()

    for config in configs:
        if time.time() > deadline - 20.0:
            break
        score, areas = _score_fast(config)
        if score > best_score:
            best_score = score
            best_points = config
        if areas:
            for area, count in sorted(areas.items(), key=lambda x: -x[1])[:5]:
                if count > 5:
                    target_areas.add(area)
    if not target_areas:
        target_areas = {1, 2, 3, 4, 5, 6, 8}

    top_configs = []
    for config in configs[:20]:
        score, _ = _score_fast(config)
        top_configs.append((score, config))
    top_configs.sort(key=lambda x: -x[0])

    # Smooth-surrogate SA from the best grid/lattice starts (no fixed target)
    for score, config in top_configs[:6]:
        if time.time() > deadline - 12.0:
            break
        remaining = deadline - time.time() - 3.0
        if remaining < 3.0:
            break
        ls_pts, _ = _local_search(config, time.time() + min(remaining * 0.9, 10.0), target_area=None)
        final_score, _ = _score_fast(ls_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = ls_pts

    # Target-area-guided SA from best configs
    for score, config in top_configs[:5]:
        if time.time() > deadline - 8.0:
            break
        _, areas = _score_fast(config)
        if not areas:
            continue
        top_area = max(areas, key=areas.get)
        remaining = deadline - time.time() - 3.0
        if remaining < 3.0:
            break
        ls_pts, _ = _local_search(config, time.time() + min(remaining * 0.9, 8.0), target_area=top_area)
        final_score, _ = _score_fast(ls_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = ls_pts

    # Target-area-guided SA from a couple of random starts
    for ta in list(target_areas)[:4]:
        if time.time() > deadline - 6.0:
            break
        remaining = deadline - time.time() - 2.0
        if remaining < 3.0:
            break
        ls_pts, _ = _local_search(best_points, time.time() + min(remaining * 0.85, 6.0), target_area=ta)
        final_score, _ = _score_fast(ls_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = ls_pts

    # Final polish with smooth surrogate
    if time.time() < deadline - 3.0:
        _, final_areas = _score_fast(best_points)
        top_area = max(final_areas, key=final_areas.get) if final_areas else None
        remaining = deadline - time.time() - 1.0
        if remaining > 2.0:
            polished, _ = _local_search(best_points, time.time() + min(remaining * 0.9, 5.0), target_area=top_area)
            polished_score, _ = _score_fast(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished

    return best_points


def construct():
    start_time = time.time()
    try:
        result = _try_all_configs()
    except Exception:
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    # Ensure all points are tuples of integers
    result = [(int(x), int(y)) for (x, y) in result]
    pts = set(result)
    if len(pts) != 30:
        result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
            break
    return result
# EVOLVE-BLOCK-END
