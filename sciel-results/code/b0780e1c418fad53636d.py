"""C4-free edge set construction for Q7: precomputed greedy + local search."""
import time
import random
from array import array

_D = 7
_N = 1 << _D
_NEDGES = _D * _N // 2  # 448

_edges = None
_edge_faces = None


def _precompute():
    d = _D
    n = _N
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
    n_edges = len(edges)
    # faces: for each edge, 6 faces, each a 3-tuple of neighbor edge indices
    faces = [None] * n_edges
    for idx, (x, y) in enumerate(edges):
        i = (x ^ y).bit_length() - 1
        fl = []
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            a, b = x, c
            if a > b:
                a, b = b, a
            f1 = edge_idx[(a, b)]
            a, b = y, e
            if a > b:
                a, b = b, a
            f2 = edge_idx[(a, b)]
            a, b = c, e
            if a > b:
                a, b = b, a
            f3 = edge_idx[(a, b)]
            fl.append((f1, f2, f3))
        faces[idx] = fl
    return edges, faces


def _get_precomputed():
    global _edges, _edge_faces
    if _edges is None:
        _edges, _edge_faces = _precompute()
    return _edges, _edge_faces


def _greedy(order, deadline):
    """Greedy in given order. Returns (selected_bool_array, conflict_array)."""
    n_edges = _NEDGES
    faces = _edge_faces
    selected = array('b', [0]) * n_edges
    conflict = array('b', [0]) * n_edges
    for idx in order:
        if time.time() > deadline:
            break
        fl = faces[idx]
        ok = True
        for a, b, c in fl:
            if selected[a] and selected[b] and selected[c]:
                ok = False
                break
        if ok:
            selected[idx] = 1
            # adding idx closes any face where the other 2 are already selected
            for a, b, c in fl:
                if selected[a] and selected[b]:
                    conflict[c] += 1
                if selected[a] and selected[c]:
                    conflict[b] += 1
                if selected[b] and selected[c]:
                    conflict[a] += 1
    return selected, conflict


def _local_search(selected, conflict, deadline, rng):
    """Local search: add conflict-free edges, then swap to close open faces."""
    n_edges = _NEDGES
    faces = _edge_faces
    edges = _edges
    sel = selected
    conf = conflict

    while time.time() < deadline:
        improved = False

        # Pass A: add any currently addable (conflict==0) edge
        for idx in range(n_edges):
            if not sel[idx] and conf[idx] == 0:
                fl = faces[idx]
                sel[idx] = 1
                for a, b, c in fl:
                    if sel[a] and sel[b]:
                        conf[c] += 1
                    if sel[a] and sel[c]:
                        conf[b] += 1
                    if sel[b] and sel[c]:
                        conf[a] += 1
                improved = True

        # Pass B: swaps. For each selected edge with an open face, try to remove
        # it and add the currently-blocked neighbor that was closing that face.
        for idx in range(n_edges):
            if time.time() > deadline:
                break
            if not sel[idx] or conf[idx] == 0:
                continue
            fl = faces[idx]
            # find an open face and its currently-selected blocking neighbor
            open_face = None
            block = -1
            for (a, b, c) in fl:
                if not (sel[a] and sel[b] and sel[c]):
                    open_face = (a, b, c)
                    # the neighbor that is selected but whose removal would open the face
                    for n in (a, b, c):
                        if sel[n] and conf[n] > 0:
                            block = n
                            break
                    break
            if open_face is None or block < 0:
                continue
            # try: remove idx, add block (block must be currently unselected)
            if sel[block]:
                continue
            # temporarily remove idx
            sel[idx] = 0
            for a, b, c in fl:
                if sel[a] and sel[b]:
                    conf[c] -= 1
                if sel[a] and sel[c]:
                    conf[b] -= 1
                if sel[b] and sel[c]:
                    conf[a] -= 1
            # can we add block now?
            flb = faces[block]
            addable = True
            for a, b, c in flb:
                if sel[a] and sel[b] and sel[c]:
                    addable = False
                    break
            if addable:
                sel[block] = 1
                for a, b, c in flb:
                    if sel[a] and sel[b]:
                        conf[c] += 1
                    if sel[a] and sel[c]:
                        conf[b] += 1
                    if sel[b] and sel[c]:
                        conf[a] += 1
                improved = True
            else:
                # revert removal of idx
                sel[idx] = 1
                for a, b, c in fl:
                    if sel[a] and sel[b]:
                        conf[c] += 1
                    if sel[a] and sel[c]:
                        conf[b] += 1
                    if sel[b] and sel[c]:
                        conf[a] += 1

        if not improved:
            break

    return sel


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _is_c4_free(edge_list):
    """Fast validation: no 2D face has all 4 edges selected."""
    n = _N
    sel = [False] * _NEDGES
    # Build index lookup
    idx_of = {}
    for x in range(n):
        for i in range(_D):
            y = x ^ (1 << i)
            if x < y:
                idx_of[(x, y)] = len(idx_of)
    for u, v in edge_list:
        key = (u, v) if u < v else (v, u)
        sel[idx_of[key]] = True
    # check faces using precomputed face structure
    faces = _edge_faces
    for idx in range(_NEDGES):
        if not sel[idx]:
            continue
        for a, b, c in faces[idx]:
            if sel[a] and sel[b] and sel[c]:
                return False
    return True


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    _get_precomputed()
    edges = _edges
    n_edges = _NEDGES
    rng = random.Random(42)

    best_sel = None
    best_size = 0

    def _record(sel):
        nonlocal best_sel, best_size
        cnt = sum(sel)
        if cnt > best_size:
            best_size = cnt
            best_sel = sel[:]

    def _order_biased(rng):
        # dimension-interleaved random order: shuffle dims, then within each dim shuffle vertices
        dim_order = list(range(_D))
        rng.shuffle(dim_order)
        order = []
        for i in dim_order:
            col = []
            for x in range(_N):
                y = x ^ (1 << i)
                if x < y:
                    # index of this edge
                    # x-major: index = x*_D//2 + (i//2) if i even else x*_D//2 + i//2
                    idx = x * _D // 2 + (i // 2)
                    col.append(idx)
            rng.shuffle(col)
            order.extend(col)
        return order

    # Phase 1: deterministic greedy baseline
    sel, conf = _greedy(list(range(n_edges)), deadline)
    sel = _local_search(sel, conf, deadline, rng)
    _record(sel)

    # Phase 2: randomized restarts (biased order) + local search
    it = 0
    while time.time() < deadline:
        order = _order_biased(rng)
        sel, conf = _greedy(order, deadline)
        sel = _local_search(sel, conf, deadline, rng)
        _record(sel)
        it += 1
        if best_size >= 308:
            break

    # Build result from best_sel
    if best_sel is None:
        # Fallback: empty set is valid
        return []
    result = []
    for idx in range(n_edges):
        if best_sel[idx]:
            u, v = edges[idx]
            result.append([u, v])

    # Validate; if somehow invalid, fall back to a clean deterministic greedy
    if result and not _is_c4_free(result):
        sel, conf = _greedy(list(range(n_edges)), deadline)
        result = []
        for idx in range(n_edges):
            if sel[idx]:
                u, v = edges[idx]
                result.append([u, v])

    return result
