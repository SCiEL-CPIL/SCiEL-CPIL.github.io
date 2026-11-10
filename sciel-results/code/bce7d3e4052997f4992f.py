import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 600.0
_MARGIN = 15.0

P = 7
K = 3
Q = P * P  # 49
N2 = Q * Q  # 2401

_SQ = {0, 1, 2, 4}


def _is_sq(a):
    return a in _SQ


def _build_complement():
    comp = set()
    for xa in range(P):
        for xb in range(P):
            for ya in range(Q):
                if not _is_sq(ya):
                    continue
                for yb in range(Q):
                    if (2 * xa + ya * ya - K * yb * yb) % P == 0:
                        comp.add((xa, xb, ya, yb))
    return comp


def _line_pts(c, d):
    ca, cb, cya, cyb = c
    pts = []
    if d == 0:
        for t in range(Q):
            pts.append((ca, cb, (cya + t) % Q, cyb))
        return pts
    ma = (d - 1) // P
    mb = (d - 1) % P
    for sa in range(P):
        for sb in range(P):
            x_a = (ca + sa) % P
            x_b = (cb + sb) % P
            sm_a = (sa * ma - sb * mb * K) % P
            sm_b = (sa * mb + sb * ma) % P
            y_a = (cya + sm_a) % P
            y_b = (cyb + sm_b) % P
            pts.append((x_a, x_b, y_a, y_b))
    return pts


def _valid_dirs(x, comp):
    xa, xb, ya, yb = x
    res = set()
    ok = True
    for t in range(Q):
        if t == ya:
            continue
        if (xa, xb, (ya + t) % Q, yb) in comp:
            ok = False
            break
    if ok:
        res.add(0)
    for ma in range(P):
        for mb in range(P):
            ok = True
            for sa in range(P):
                for sb in range(P):
                    if sa == xa and sb == xb:
                        continue
                    x_a = (xa + sa) % P
                    x_b = (xb + sb) % P
                    sm_a = (sa * ma - sb * mb * K) % P
                    sm_b = (sa * mb + sb * ma) % P
                    y_a = (ya + sm_a) % P
                    y_b = (yb + sm_b) % P
                    if (x_a, x_b, y_a, y_b) in comp:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                res.add(1 + ma * P + mb)
    return res


def nikodym_set() -> np.ndarray:
    comp = _build_complement()

    all_pts = []
    for xa in range(P):
        for xb in range(P):
            for ya in range(Q):
                for yb in range(Q):
                    all_pts.append((xa, xb, ya, yb))

    # Precompute valid dirs
    vdirs = {x: _valid_dirs(x, comp) for x in all_pts}

    candidates = [x for x in all_pts if x not in comp]
    # Limit greedy additions to keep within time
    max_add = 30
    added = 0
    for x in candidates:
        if added >= max_add or time.time() > _DEADLINE - _MARGIN:
            break
        if x in comp:
            continue
        comp.add(x)
        affected = set()
        for d in range(49):
            for pt in _line_pts(x, d):
                if pt != x:
                    affected.add(pt)
        ok = True
        for pt in affected:
            if len(vdirs[pt]) <= 1:
                nv = _valid_dirs(pt, comp)
                if len(nv) == 0:
                    ok = False
                    break
                vdirs[pt] = nv
        if ok:
            added += 1
        else:
            comp.remove(x)
            for pt in affected:
                vdirs[pt] = _valid_dirs(pt, comp)

    points = []
    for xa in range(P):
        for xb in range(P):
            for ya in range(Q):
                for yb in range(Q):
                    if (xa, xb, ya, yb) not in comp:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
