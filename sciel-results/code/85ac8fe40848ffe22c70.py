"""Constraint-based backtracking snake solver for Q_9.

Representation:
  - path: list of ints (snake vertices in order)
  - used_mask: Python int bitset, bit i set iff vertex i is in the path
  - forbidden[v]: precomputed bitset of all vertices at Hamming distance 1 or 2 from v.
    A candidate w is valid iff (used_mask & forbidden[w]) == 0 and w not in used_mask.

Search: depth-first backtracking with minimum-remaining-values (MRV) heuristic.
At each node we score each candidate extension by how many future extensions it
leaves for the new tail and pick the most constrained first, pruning dead-ends
early. We keep the longest valid path found and respect a wall-clock deadline
with a safety margin. Multiple restarts with different bit-order tie-breaks
provide diversity to escape local optima.
"""

import time
import random

N = 9
M = 1 << N  # 512 vertices

# Precompute forbidden bitsets: forbidden[v] has bit u set iff 1 <= popcount(v^u) <= 2.
_FORBIDDEN = [0] * M
for _v in range(M):
    _m = 0
    for _u in range(M):
        _d = (_v ^ _u).bit_count()
        if _d == 1 or _d == 2:
            _m |= (1 << _u)
    _FORBIDDEN[_v] = _m

# Precompute neighbor lists (distance 1).
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(M)]

_DEADLINE = 150.0
_SAFETY = 12.0


def _greedy_seed():
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _mask_of(path):
    m = 0
    for v in path:
        m |= (1 << v)
    return m


def _bit_orders():
    base = list(range(N))
    yield base
    yield list(reversed(base))
    rng = random.Random(12345)
    seen = {tuple(base), tuple(reversed(base))}
    for _ in range(12):
        perm = base[:]
        rng.shuffle(perm)
        if tuple(perm) not in seen:
            seen.add(tuple(perm))
            yield perm


def _count_future_options(tail, used_mask):
    fm = _FORBIDDEN
    um = used_mask
    cnt = 0
    for w in _NEIGHBORS[tail]:
        if (um >> w) & 1:
            continue
        if um & fm[w]:
            continue
        cnt += 1
    return cnt


def _run_backtrack(seed_path, bit_order, t0):
    """DFS backtracking with MRV, seeded from seed_path. Returns best path found."""
    best = list(seed_path)
    best_len = len(best) - 1
    fm = _FORBIDDEN
    neigh = _NEIGHBORS
    order = bit_order

    def time_left():
        return _DEADLINE - _SAFETY - (time.time() - t0)

    def search(path, used_mask):
        nonlocal best, best_len
        if time_left() <= 0:
            return
        cur = path[-1]
        cur_len = len(path) - 1
        if cur_len > best_len:
            best_len = cur_len
            best = list(path)

        um = used_mask
        candidates = []
        for b in order:
            w = cur ^ (1 << b)
            if (um >> w) & 1:
                continue
            if um & fm[w]:
                continue
            candidates.append(w)

        if not candidates:
            return

        scored = []
        for w in candidates:
            new_um = um | (1 << w)
            fut = _count_future_options(w, new_um)
            scored.append((fut, w, new_um))
        scored.sort(key=lambda x: x[0])

        for fut, w, new_um in scored:
            if time_left() <= 0:
                return
            path.append(w)
            search(path, new_um)
            path.pop()

    search(list(seed_path), _mask_of(seed_path))
    return best


def construct():
    start = time.time()
    best_path = _greedy_seed()
    best_len = len(best_path) - 1

    for bit_order in _bit_orders():
        if _DEADLINE - _SAFETY - (time.time() - start) <= 0:
            break
        path = _run_backtrack(best_path, bit_order, start)
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path

    return best_path
