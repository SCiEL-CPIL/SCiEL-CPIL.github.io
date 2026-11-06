import time
import random

_D = 7
_N = 128
_EDGES = 448
_DEADLINE = 120.0
_SAFETY = 6.0


def _build_tables():
    edge_list = []
    edge_index = {}
    for x in range(_N):
        for i in range(_D):
            y = x ^ (1 << i)
            if x < y:
                edge_index[(x, y)] = len(edge_list)
                edge_list.append((x, y))
    assert len(edge_list) == _EDGES

    # partner_faces[k] = list of 3-tuples, one per face touching edge k,
    # each tuple = the 3 other edges of that face.
    partner_faces = [None] * _EDGES
    for k, (x, y) in enumerate(edge_list):
        i = (x ^ y).bit_length() - 1
        faces = []
        for j in range(_D):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            faces.append((
                edge_index[(min(x, c), max(x, c))],
                edge_index[(min(y, e), max(y, e))],
                edge_index[(min(c, e), max(c, e))],
            ))
        partner_faces[k] = faces
    return edge_list, edge_index, partner_faces


_EDGE_LIST, _EDGE_INDEX, _PARTNER_FACES = _build_tables()


def _is_safe(k, present):
    for a, b, c in _PARTNER_FACES[k]:
        if present[a] and present[b] and present[c]:
            return False
    return True


def _greedy_seed(rnd, deadline):
    present = bytearray(_EDGES)
    count = 0
    order = list(range(_EDGES))
    rnd.shuffle(order)
    for k in order:
        if time.time() > deadline:
            break
        if _is_safe(k, present):
            present[k] = 1
            count += 1
    return present, count


def _topup(present, count, deadline):
    # repeatedly insert any currently-safe edge (1-for-1) until none remain
    while True:
        improved = False
        for k in range(_EDGES):
            if present[k]:
                continue
            if _is_safe(k, present):
                present[k] = 1
                count += 1
                improved = True
        if not improved or time.time() > deadline:
            break
    return count


def _local_search(present, count, deadline, rnd):
    count = _topup(present, count, deadline)

    # 2-for-1 / 3-for-1: for a blocked edge k, remove one present partner q
    # (from a face of k that is 2/3 complete) so k becomes safe, insert k,
    # then top-up; keep the change only if the net count increased.
    blocked = [k for k in range(_EDGES) if not present[k]]
    for _round in range(400):
        if time.time() > deadline:
            break
        rnd.shuffle(blocked)
        progress = False
        for k in blocked:
            if present[k]:
                continue
            faces = _PARTNER_FACES[k]
            for (p0, p1, p2) in faces:
                pp = [p for p in (p0, p1, p2) if present[p]]
                if len(pp) != 2:
                    continue
                for q in pp:
                    if time.time() > deadline:
                        break
                    present[q] = 0
                    if _is_safe(k, present):
                        present[k] = 1
                        cnt2 = count - 1 + 1  # -q +k
                        cnt2 = _topup(present, cnt2, deadline)
                        if cnt2 > count:
                            count = cnt2
                            progress = True
                            if k in blocked:
                                blocked.remove(k)
                            break
                    else:
                        present[q] = 1
        if not progress:
            break
    return count


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _SAFETY
    rnd = random.Random(20240517)

    best_present = None
    best_count = 0

    for s in range(8):
        if time.time() > deadline:
            break
        present, count = _greedy_seed(rnd, deadline)
        count = _local_search(present, count, deadline, rnd)
        if count > best_count:
            best_count = count
            best_present = present.copy()

    if best_present is not None and time.time() < deadline - 0.5:
        present = best_present.copy()
        count = best_count
        count = _local_search(present, count, deadline, rnd)
        if count > best_count:
            best_count = count
            best_present = present

    out = [[_EDGE_LIST[k][0], _EDGE_LIST[k][1]] for k in range(_EDGES) if best_present[k]]
    return out
