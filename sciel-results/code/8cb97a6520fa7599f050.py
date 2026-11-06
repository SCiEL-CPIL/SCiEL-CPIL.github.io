"""Grid baseline + incremental simulated annealing to maximize equal-area multiplicity."""
import time
import random
import numpy as np

_DEADLINE = 78.0          # safety margin under the 90 s kill
_BOX = 30
_N = 30

# ---- Pre-computed triple bookkeeping (30 choose 3 = 4500) ----
_iA, _jA, _kA = np.indices((_N, _N, _N))
_mask = (_iA < _jA) & (_jA < _kA)
_TI = _iA[_mask].astype(np.int32)
_TJ = _jA[_mask].astype(np.int32)
_TK = _kA[_mask].astype(np.int32)
_NTRI = len(_TI)
_INV = [np.flatnonzero((_TI == i) | (_TJ == i) | (_TK == i)) for i in range(_N)]


def _recompute(pts, idx):
    """Recompute areas for all triples involving point idx (in place)."""
    m = _INV[idx]
    pi = pts[_TI[m]]; pj = pts[_TJ[m]]; pk = pts[_TK[m]]
    dets = (pi[:, 0] * (pj[:, 1] - pk[:, 1]) +
            pj[:, 0] * (pk[:, 1] - pi[:, 1]) +
            pk[:, 0] * (pi[:, 1] - pj[:, 1]))
    return np.abs(dets).astype(np.int64)


def _full_areas(pts):
    a = np.empty(_NTRI, dtype=np.int64)
    for i in range(_N):
        a[_INV[i]] = _recompute(pts, i)
    return a


def _best(areas):
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0, 0
    vals, counts = np.unique(pos, return_counts=True)
    mi = int(np.argmax(counts))
    return int(counts[mi]), int(vals[mi])


def _sa(pts0, deadline, target=None):
    """Simulated annealing with incremental updates + dynamic target switching."""
    pts = np.array(pts0, dtype=np.int64)
    occ = set(map(tuple, pts.tolist()))
    areas = _full_areas(pts)

    cm, ca = _best(areas)
    if target is None:
        target = ca
    tc = int(np.sum(areas == target))
    score = max(tc, cm)

    best = score
    best_pts = pts.copy()

    temp = 2.0
    cool = 0.99985
    tmin = 0.002
    steps = 0
    last = 0
    RINT = 2000

    while time.time() < deadline:
        steps += 1
        idx = random.randrange(_N)
        ox, oy = int(pts[idx, 0]), int(pts[idx, 1])

        r = random.random()
        if r < 0.30:
            nx = random.randint(-_BOX, _BOX)
            ny = random.randint(-_BOX, _BOX)
        elif r < 0.70:
            nx = ox + random.randint(-3, 3)
            ny = oy + random.randint(-3, 3)
        else:
            nx = ox + random.randint(-8, 8)
            ny = oy + random.randint(-8, 8)
        nx = max(-_BOX, min(_BOX, nx))
        ny = max(-_BOX, min(_BOX, ny))
        if (nx, ny) == (ox, oy) or (nx, ny) in occ:
            continue

        pts[idx] = (nx, ny)
        occ.discard((ox, oy)); occ.add((nx, ny))

        areas[_INV[idx]] = _recompute(pts, idx)
        if steps - last >= RINT:
            areas = _full_areas(pts)
            last = steps

        nm, na = _best(areas)
        if na != target:
            target = na
        nt = int(np.sum(areas == target))
        ns = max(nt, nm)

        if ns > score:
            accept = True
        elif ns == score:
            accept = (nt > tc) or (random.random() < 0.4)
        else:
            accept = random.random() < np.exp(-(score - ns) / temp)

        if accept:
            score, cm, ca, tc, target = ns, nm, na, nt, target
            if score > best:
                best = score
                best_pts = pts.copy()
            temp = max(tmin, temp * cool)
        else:
            pts[idx] = (ox, oy)
            occ.discard((nx, ny)); occ.add((ox, oy))
            areas[_INV[idx]] = _recompute(pts, idx)
            temp = max(tmin, temp * cool)
    return best, best_pts


def _sa_multi(pts0, deadline):
    """SA with a restart on stagnation."""
    best = -1
    best_pts = np.array(pts0, dtype=np.int64)
    cur = np.array(pts0, dtype=np.int64)
    steps = 0
    last = 0
    RINT = 2000
    while time.time() < deadline:
        steps += 1
        idx = random.randrange(_N)
        ox, oy = int(cur[idx, 0]), int(cur[idx, 1])
        r = random.random()
        if r < 0.30:
            nx = random.randint(-_BOX, _BOX); ny = random.randint(-_BOX, _BOX)
        elif r < 0.70:
            nx = ox + random.randint(-3, 3); ny = oy + random.randint(-3, 3)
        else:
            nx = ox + random.randint(-8, 8); ny = oy + random.randint(-8, 8)
        nx = max(-_BOX, min(_BOX, nx)); ny = max(-_BOX, min(_BOX, ny))
        if (nx, ny) == (ox, oy):
            continue
        # occupancy check via set
        curset = set(map(tuple, cur.tolist()))
        if (nx, ny) in curset:
            continue
        cur[idx] = (nx, ny)
        areas = _full_areas(cur)   # simpler full recompute for this variant
        cm, ca = _best(areas)
        tc = int(np.sum(areas == ca))
        score = max(tc, cm)
        if score > best:
            best = score
            best_pts = cur.copy()
            last = steps
        if steps - last > 4000:
            cur = best_pts.copy()
            last = steps
    return best, best_pts


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    random.seed(12345)

    # Baseline: 6x5 grid.
    base = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    base = np.array(base, dtype=np.int64)
    bv, _ = _sa(base, start + 6.0)
    best_val = bv
    best_pts = base.copy()

    # Main SA run from the grid.
    v, p = _sa(base, deadline - 12.0)
    if v > best_val:
        best_val, best_pts = v, p

    # Alternative seeds if time remains.
    if time.time() < deadline - 12.0:
        seed2 = [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]
        v2, p2 = _sa(np.array(seed2, dtype=np.int64), deadline - 6.0)
        if v2 > best_val:
            best_val, best_pts = v2, p2

    if time.time() < deadline - 6.0:
        rnd = np.array(_distinct_random(), dtype=np.int64)
        v3, p3 = _sa(rnd, deadline - 2.0)
        if v3 > best_val:
            best_val, best_pts = v3, p3

    # Final guard: exactly 30 distinct in-range points.
    seen = set(); out = []
    for p in best_pts:
        x, y = int(p[0]), int(p[1])
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y)); out.append((x, y))
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y)); out.append((x, y))
            if len(out) >= 30:
                break
    return out[:30]


def _distinct_random():
    pts = set()
    while len(pts) < _N:
        pts.add((random.randint(-_BOX, _BOX), random.randint(-_BOX, _BOX)))
    return list(pts)
