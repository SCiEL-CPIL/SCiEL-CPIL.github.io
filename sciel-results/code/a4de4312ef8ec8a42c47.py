# EVOLVE-BLOCK-START
import time
import numpy as np

_p = 7
_k = 3  # alpha^2 = 3 (quadratic non-residue mod 7)
_q = 49
_NP = 2401  # q^2 points
_NL = 50 * 49  # 50 directions * 49 points each


def _fadd(a, b):
    return ((a[0] + b[0]) % _p, (a[1] + b[1]) % _p)


def _fmul(a, b):
    a0, a1 = a
    b0, b1 = b
    return ((a0 * b0 + a1 * b1 * _k) % _p, (a0 * b1 + a1 * b0) % _p)


def _t_mul(t, e):
    """Scalar t in F_q (as index) times field element e (as pair)."""
    return _fmul((t // _p, t % _p), e)


def _build():
    p = _p
    k = _k
    q = _q
    NP = _NP
    NL = _NL

    # ---- all 50 directions ----
    dirs = [(1, m) for m in range(q)] + [(0, 1)]

    # ---- line membership: for each point, list of (line_id, t) ----
    # line_id = d*q + c ; point on line = c + t*dir
    point_lines = [[] for _ in range(NP)]
    for d in range(NL // q):
        dv = dirs[d]
        for c in range(q):
            cv = (c // p, c % p)
            for t in range(q):
                tv = _t_mul(t, (1, 0))  # t as scalar
                pt = _fadd(cv, _fmul(tv, dv))
                point_lines[pt[0] * p + pt[1]].append((d * q + c, t))

    # ---- unital U: 2*xa + ya^2 - yb^2*k = 0 ----
    in_U = np.zeros(NP, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        in_U[(xa * p + xb) * q + (ya * p + yb)] = True

    # ---- greedy maximal exclusion: prefer points with fewer lines ----
    cand = [i for i in range(NP) if in_U[i]]
    cand.sort(key=lambda i: len(point_lines[i]))
    excluded = set()
    for i in cand:
        ok = True
        for (lid, t) in point_lines[i]:
            if t == 0:
                continue
            other = (lid // q) * q + (lid % q + t) % q
            if other not in excluded:
                ok = False
                break
        if ok:
            excluded.add(i)

    return np.array(sorted(excluded), dtype=np.int64)


def _fallback():
    p = _p
    k = _k
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    out = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0 and (ya, yb) not in y_special:
                        out.append((xa * p + xb) * _q + (ya * p + yb))
    return np.array(sorted(out), dtype=np.int64)


def _valid(excluded) -> bool:
    """Every non-excluded point must have a punctured line fully in N."""
    p = _p
    q = _q
    NP = _NP
    ex = set(int(x) for x in excluded)
    if len(ex) != len(excluded):
        return False
    dirs = [(1, m) for m in range(q)] + [(0, 1)]
    for x in range(NP):
        if x in ex:
            continue
        xa, xb = divmod(x // q, p)
        ya, yb = divmod(x % q, p)
        found = False
        for dv in dirs:
            vx, vy = dv
            ok = True
            for t in range(1, q):
                tv = _t_mul(t, (1, 0))
                nx = _fadd((xa, xb), _fmul(tv, vx))
                ny = _fadd((ya, yb), _fmul(tv, vy))
                if (nx[0] * p + nx[1]) * q + (ny[0] * p + ny[1]) in ex:
                    ok = False
                    break
            if ok:
                found = True
                break
        if not found:
            return False
    return True


def _to_points(excluded) -> np.ndarray:
    p = _p
    q = _q
    pts = []
    for x in excluded:
        x = int(x) % (q * q)
        xa, xb = divmod(x // q, p)
        ya, yb = divmod(x % q, p)
        pts.append([[xa, xb], [ya, yb]])
    return np.array(pts, dtype=np.int64)


def nikodym_set() -> np.ndarray:
    t0 = time.time()
    try:
        ex = _build()
    except Exception:
        ex = _fallback()
    if time.time() - t0 > 580.0:
        ex = _fallback()
    if not _valid(ex):
        ex = _fallback()
    return _to_points(ex)


# EVOLVE-BLOCK-END
