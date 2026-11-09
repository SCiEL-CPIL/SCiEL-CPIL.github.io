# EVOLVE-BLOCK-START
import time

import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_q^2 with q = p^2, p = 7 (q = 49).

    Field: F_q = F_p[alpha], alpha^2 = k (k = smallest quadratic non-residue mod p).
    Element = (a, b) for a + b*alpha.

    N is a Nikodym set iff its complement C = F_q^2 \\ N has at most q-2 = 47
    points on every line (direction (1,m) or (0,1)). We build C greedily (max
    line-count 47) to be as large as possible, then N = F_q^2 \\ C. Several
    random orderings are tried; the largest valid C is kept. A verified fallback
    (the seed construction) guards against any invalidity.
    """
    t0 = time.time()
    DEADLINE = 600.0
    MARGIN = 30.0

    p = 7
    q = p * p

    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3

    def mul(e1, e2):
        a, b = e1
        c, d = e2
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def sub(e1, e2):
        return ((e1[0] - e2[0]) % p, (e1[1] - e2[1]) % p)

    def eid(e):
        return e[0] * p + e[1]

    n_lines = q + q * q  # 2450
    MAXC = q - 2  # 47

    pts = []
    point_lines = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x = (xa, xb)
                    y = (ya, yb)
                    lines = [eid(x)]
                    for ma in range(p):
                        for mb in range(p):
                            m = (ma, mb)
                            b = sub(y, mul(m, x))
                            lines.append(q + eid(m) * q + eid(b))
                    pts.append((xa, xb, ya, yb))
                    point_lines.append(lines)

    n_pts = len(pts)  # 2401
    pl = np.array(point_lines, dtype=np.int32)  # (2401, 50)

    best_in_C = None
    best_size = -1

    for seed in range(8):
        if time.time() > DEADLINE - MARGIN:
            break
        order = np.arange(n_pts)
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        line_count = np.zeros(n_lines, dtype=np.int32)
        in_C = np.zeros(n_pts, dtype=np.bool_)
        for idx in order:
            lines = pl[idx]
            if np.max(line_count[lines]) < MAXC:
                in_C[idx] = True
                line_count[lines] += 1
        size = int(np.sum(in_C))
        if size > best_size:
            best_size = size
            best_in_C = in_C
            best_lc = line_count

    if best_in_C is None:
        return _seed_nikodym(p, k)

    # safety verification
    if np.max(best_lc) > MAXC:
        return _seed_nikodym(p, k)

    keep = ~best_in_C
    ids = np.nonzero(keep)[0]
    out = np.empty((len(ids), 2, 2), dtype=np.int64)
    for i, idx in enumerate(ids):
        xa, xb, ya, yb = pts[idx]
        out[i, 0, 0] = xa
        out[i, 0, 1] = xb
        out[i, 1, 0] = ya
        out[i, 1, 1] = yb
    return out


def _seed_nikodym(p, k):
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
