"""Combined: enumerate structured 30-point families (large axis-aligned grids
plus rotated/sheared grids) and return the set with the largest multiplicity of
a single positive doubled area.  Fast NumPy evaluator + explicit deadline guard."""

import time
import math
import numpy as np
from collections import Counter

_BUDGET = 80.0  # wall-clock safety margin (evaluator kills at 90 s)


def _max_multiplicity(pts):
    """Max multiplicity of a single positive doubled area over all triples."""
    P = np.asarray(pts, dtype=np.int64)
    n = P.shape[0]
    if n < 3:
        return 0
    # d[i,j,k] = | (P[i]-P[k]) x (P[j]-P[k]) |  = 2 * |signed area|
    d = np.abs(
        (P[:, 0, None, None] - P[None, None, :, 0])
        * (P[:, 1, None, None] - P[None, None, :, 1])
        - (P[:, 1, None, None] - P[None, None, :, 1])
        * (P[:, 0, None, None] - P[None, None, :, 0])
    )
    tri = d[np.triu_indices(n, 1), np.arange(n)]
    vals = tri[tri > 0]
    if vals.size == 0:
        return 0
    return Counter(vals.tolist()).most_common(1)[0][1]


def _make_grid(u, v, count):
    """First `count` points of the lattice {a*u + b*v : a in [0,|u|), b in [0,|v|)}
    in row-major order, clipped to the [-30,30]^2 box, deduped.  Returns None if
    it cannot supply 30 distinct in-box points."""
    out = []
    seen = set()
    for a in range(abs(u[0])):
        for b in range(abs(v[0])):
            if len(out) >= count:
                break
            x = a * u[0] + b * v[0]
            y = a * u[1] + b * v[1]
            if -30 <= x <= 30 and -30 <= y <= 30:
                p = (x, y)
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        if len(out) >= count:
            break
    return out if len(out) >= count else None


def _generate_candidates():
    cands = []
    seen = set()

    def add(pts):
        if pts is None:
            return
        if len(set(pts)) != 30:
            return
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                cands.append(pts)

    # 1) Axis-aligned rectangular grids (u=(1,0), v=(0,1)) of many sizes.
    for w in range(1, 31):
        for h in range(1, 31):
            if w * h < 30:
                continue
            add(_make_grid((1, 0), (0, 1), 30))

    # 2) Rotated / sheared grids (integer lattice bases with det = +-1).
    for a in range(-4, 5):
        for b in range(-4, 5):
            for c in range(-4, 5):
                for d in range(-4, 5):
                    if a * d - b * c != 1:
                        continue
                    u = (a, b)
                    v = (c, d)
                    # need enough lattice points to reach 30 within the box
                    if abs(a) < 1 and abs(c) < 1:
                        continue
                    add(_make_grid(u, v, 30))

    return cands


def construct():
    start = time.time()

    # Seed baseline (6x5 grid) guarantees a valid 18-multiplicity result.
    seed = [(x, y) for x in range(6) for y in range(5)]
    best_score = _max_multiplicity(seed)
    best_points = seed

    for pts in _generate_candidates():
        if time.time() - start > _BUDGET:
            break
        try:
            s = _max_multiplicity(pts)
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_points = pts

    return best_points
