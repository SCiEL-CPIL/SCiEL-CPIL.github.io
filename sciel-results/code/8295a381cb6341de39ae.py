"""Rewritten program for C4-free edge set on Q7 using hitting-set / removal formulation."""
import time
import random

# Precompute all edges and all 2-faces of Q7
_D = 7
_N = 1 << _D
_TOTAL_EDGES = _D * (1 << (_D - 1))  # 448

# Build edge list with canonical ordering (u < v)
_EDGE_LIST = []
_EDGE_INDEX = {}
for _x in range(_N):
    for _i in range(_D):
        _y = _x ^ (1 << _i)
        if _x < _y:
            _e = (_x, _y)
            _EDGE_LIST.append(_e)
            _EDGE_INDEX[_e] = len(_EDGE_LIST) - 1

# Build all 2-faces: each face is defined by vertex x and two distinct dimensions i < j
# The four edges of the face are:
#   (x, x^e_i), (x^e_i, x^e_i^e_j), (x^e_j, x^e_i^e_j), (x, x^e_j)
# where e_i = 1<<i, e_j = 1<<j
_FACES = []  # list of tuples of 4 edge indices
for _x in range(_N):
    for _i in range(_D):
        for _j in range(_i + 1, _D):
            _ei = 1 << _i
            _ej = 1 << _j
            _v0 = _x
            _v1 = _x ^ _ei
            _v2 = _x ^ _ej
            _v3 = _x ^ _ei ^ _ej
            # Four edges: (v0,v1), (v1,v3), (v3,v2), (v2,v0)
            _edges = [
                (min(_v0, _v1), max(_v0, _v1)),
                (min(_v1, _v3), max(_v1, _v3)),
                (min(_v3, _v2), max(_v3, _v2)),
                (min(_v2, _v0), max(_v2, _v0)),
            ]
            _face = tuple(_EDGE_INDEX[_e] for _e in _edges)
            _FACES.append(_face)

# Build for each edge, the list of faces it belongs to
_EDGE_TO_FACES = [[] for _ in range(_TOTAL_EDGES)]
for _fi, _face in enumerate(_FACES):
    for _ei in _face:
        _EDGE_TO_FACES[_ei].append(_fi)


def _face_complete(face, removed):
    """Check if all 4 edges of a face are NOT removed (i.e., all present)."""
    for _ei in face:
        if _ei in removed:
            return False
    return True


def _count_bad_faces(removed):
    """Count faces where all 4 edges are present."""
    cnt = 0
    for _face in _FACES:
        if _face_complete(_face, removed):
            cnt += 1
    return cnt


def _greedy_hit(removed):
    """
    Greedy hitting set: while there exists a bad face (all 4 edges present),
    pick the edge in that face whose removal hits the most bad faces.
    """
    while True:
        # Find all bad faces
        bad_faces = [f for f in _FACES if _face_complete(f, removed)]
        if not bad_faces:
            break
        # For each edge not yet removed, count how many bad faces it hits
        # Only consider edges that are in at least one bad face
        edge_hit_count = {}
        for f in bad_faces:
            for ei in f:
                if ei not in removed:
                    edge_hit_count[ei] = edge_hit_count.get(ei, 0) + 1
        if not edge_hit_count:
            break
        # Pick the edge with max hit count (tie-break by smallest index)
        best_edge = min(edge_hit_count.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        removed.add(best_edge)


def _remove_redundant(removed):
    """Remove edges from 'removed' that are not needed (i.e., removing them still leaves all faces hit)."""
    # Try removing each edge from removed and check if any face becomes bad
    changed = True
    while changed:
        changed = False
        for ei in list(removed):
            removed.discard(ei)
            # Check if any face is now bad
            bad = False
            for f in _EDGE_TO_FACES[ei]:
                if _face_complete(f, removed):
                    bad = True
                    break
            if bad:
                removed.add(ei)
            else:
                changed = True
    return removed


def _swap_improve(removed, max_iters=5000):
    """
    Local search: try swapping a removed edge with a present edge to reduce total removed count,
    or try to remove an edge from 'removed' if it's redundant (already handled),
    or try adding an edge to 'removed' only if it allows removing more.
    Here we focus on: can we remove an edge from 'removed' and add another edge from 'removed'
    to reduce the set? Actually, simpler: try to find an edge in 'removed' that can be swapped
    out for nothing (redundancy) - already done. Instead, try: for each edge in removed, see if
    we can un-remove it and remove a different edge to maintain validity with same or fewer removals.
    """
    # Try: for each edge r in removed, try to un-remove it. Then we need to hit any newly-bad faces
    # by removing other edges. If we can do it with <= 1 additional removals, net gain.
    for _ in range(max_iters):
        improved = False
        for r in list(removed):
            # Temporarily un-remove r
            removed.discard(r)
            # Find bad faces
            bad = [f for f in _EDGE_TO_FACES[r] if _face_complete(f, removed)]
            if not bad:
                improved = True
                continue
            # Try to fix by removing one edge from the bad faces
            for f in bad:
                for ei in f:
                    if ei not in removed and ei != r:
                        # Try removing ei instead
                        removed.add(ei)
                        # Check if all bad faces are now hit
                        ok = True
                        for bf in bad:
                            if _face_complete(bf, removed):
                                ok = False
                                break
                        if ok:
                            # Net: removed r, added ei -> same count, but maybe better structure
                            # Actually same count, so no improvement in size. Skip.
                            pass
                        else:
                            removed.discard(ei)
                # Re-add r since we didn't find a valid single-swap
            removed.add(r)
        if not improved:
            break
    return removed


def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    best_removed = None
    best_count = _TOTAL_EDGES  # start with no removals (invalid)

    # Phase 1: Greedy hitting set from scratch
    removed = set()
    _greedy_hit(removed)
    _remove_redundant(removed)
    count = _TOTAL_EDGES - len(removed)
    if count > best_count:
        best_count = count
        best_removed = set(removed)

    if time.time() > deadline:
        return [[u, v] for (u, v) in _EDGE_LIST if _EDGE_LIST.index((u, v)) not in best_removed]

    # Phase 2: Multiple random restarts with randomized greedy
    rng = random.Random(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        removed = set()
        # Randomized greedy: shuffle edges, but use priority based on face count with noise
        while True:
            bad_faces = [f for f in _FACES if _face_complete(f, removed)]
            if not bad_faces:
                break
            edge_hit_count = {}
            for f in bad_faces:
                for ei in f:
                    if ei not in removed:
                        edge_hit_count[ei] = edge_hit_count.get(ei, 0) + 1
            if not edge_hit_count:
                break
            # Add noise to hit counts
            noisy = {ei: hc + rng.random() * 2 for ei, hc in edge_hit_count.items()}
            best_edge = max(noisy, key=lambda ei: noisy[ei])
            removed.add(best_edge)

        _remove_redundant(removed)
        count = _TOTAL_EDGES - len(removed)
        if count > best_count:
            best_count = count
            best_removed = set(removed)

    # Phase 3: Local search / swap improvement on best solution
    if best_removed is not None and time.time() < deadline:
        best_removed = _swap_improve(best_removed, max_iters=2000)
        _remove_redundant(best_removed)
        best_count = _TOTAL_EDGES - len(best_removed)

    # Build result
    removed_set = best_removed if best_removed is not None else set()
    result = []
    for idx, (u, v) in enumerate(_EDGE_LIST):
        if idx not in removed_set:
            result.append([u, v])
    return result
