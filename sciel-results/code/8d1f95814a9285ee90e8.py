import time
import random
import numpy as np

_D = 7
_N = 1 << _D
_TIME_LIMIT = 115.0  # safety margin under 120s / 150s kill

def _build_edges():
    """All 448 undirected edges of Q7 as (u,v) with u<v."""
    edges = []
    for u in range(_N):
        for i in range(_D):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
    return edges

def _edge_index():
    """Map (u,v) with u<v -> index."""
    m = {}
    for idx, (u, v) in enumerate(_build_edges()):
        m[(u, v)] = idx
    return m

_EDGE_LIST = _build_edges()
_EID = _edge_index()
_M = len(_EDGE_LIST)  # 448

def _faces():
    """For each edge, the list of faces (as edge-index triples) that contain it.
    A face = 4-cycle: edges e, e1, e2, e3. For edge e=(u,v) differing in bit i,
    for each other bit j: c=u^(1<<j), d=v^(1<<j). The face edges are e,
    (u,c), (v,d), (c,d)."""
    faces = [[] for _ in range(_M)]
    for idx, (u, v) in enumerate(_EDGE_LIST):
        # find bit i
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(_D):
            if j == i:
                continue
            c = u ^ (1 << j)
            d = v ^ (1 << j)
            e1 = _EID[(min(u, c), max(u, c))]
            e2 = _EID[(min(v, d), max(v, d))]
            e3 = _EID[(min(c, d), max(c, d))]
            faces[idx].append((e1, e2, e3))
    return faces

_FACES = _faces()

def _count_faces(sel):
    """Number of 2-faces fully selected. sel is a boolean array of length M."""
    cnt = 0
    for idx in np.nonzero(sel)[0]:
        for (a, b, c) in _FACES[idx]:
            if sel[a] and sel[b] and sel[c]:
                cnt += 1
    return cnt

def _greedy(order, rng):
    """Greedy add edges in given order, keeping if it doesn't complete a face."""
    sel = np.zeros(_M, dtype=bool)
    chosen = set()
    for idx in order:
        ok = True
        for (a, b, c) in _FACES[idx]:
            if sel[a] and sel[b] and sel[c]:
                ok = False
                break
        if ok:
            sel[idx] = True
            chosen.add(idx)
    return sel, chosen

def _local_search(sel, rng, deadline):
    """Try swaps: remove one edge, add one blocked edge that becomes valid.
    Also 2-opt. Returns improved sel."""
    M = _M
    while time.time() < deadline:
        improved = False
        # candidate edges to add: currently not selected
        notsel = [i for i in range(M) if not sel[i]]
        rng.shuffle(notsel)
        for f in notsel:
            if time.time() >= deadline:
                break
            # faces that f would complete
            blocking = []
            for (a, b, c) in _FACES[f]:
                if sel[a] and sel[b] and sel[c]:
                    blocking.append((a, b, c))
            if not blocking:
                # can add directly
                sel[f] = True
                improved = True
                continue
            # try to remove one edge from a blocking face to allow f
            for (a, b, c) in blocking:
                for rem in (a, b, c):
                    if not sel[rem]:
                        continue
                    sel[rem] = False
                    # check f now addable
                    ok = True
                    for (x, y, z) in _FACES[f]:
                        if sel[x] and sel[y] and sel[z]:
                            ok = False
                            break
                    if ok:
                        sel[f] = True
                        improved = True
                        break
                    else:
                        sel[rem] = True
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return sel

def construct():
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT
    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    best_sel = None
    best_cnt = -1

    # Several randomized greedy orders
    base = list(range(_M))
    for restart in range(2000):
        if time.time() > deadline - 5:
            break
        # random ordering with a bit of structure: shuffle
        order = base[:]
        rng.shuffle(order)
        sel, _ = _greedy(order, rng)
        cnt = int(sel.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_sel = sel.copy()

    # Local search polish on the best
    if best_sel is not None:
        sel = best_sel.copy()
        sel = _local_search(sel, rng, deadline)
        cnt = int(sel.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_sel = sel

    if best_sel is None:
        # fallback: deterministic greedy
        sel, _ = _greedy(list(range(_M)), rng)
        best_sel = sel

    out = []
    for idx in np.nonzero(best_sel)[0]:
        u, v = _EDGE_LIST[idx]
        out.append([u, v])
    return out
