"""Snake-in-the-box in Q_9 via randomized backtracking search with bitmask pruning."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N
SEED = 104.0
_DEADLINE = 150.0 - 10.0


def _valid_snake(path):
    used = set()
    for i, v in enumerate(path):
        if v in used:
            return False
        used.add(v)
        if i > 0:
            if (v ^ path[i - 1]).bit_count() != 1:
                return False
            for j in range(i - 1):
                if (v ^ path[j]).bit_count() == 1:
                    return False
    return True


def _greedy(order):
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path


def _search(order, start, deadline, best_ref):
    path = [start]
    used = np.zeros(SIZE, dtype=bool)
    used[start] = True
    # blocked[v] = True if v is adjacent to any forbidden path vertex
    # Forbidden = all path vertices except the last one
    # Initially only path[0] exists, it's the last, so nothing is forbidden yet.
    # But wait: when path has length 1, there are no non-consecutive pairs, so nothing blocked.
    # When we add path[1], path[0] becomes forbidden (non-consecutive to future path[2+]).
    blocked = np.zeros(SIZE, dtype=bool)
    best_len = best_ref[0]

    def backtrack():
        nonlocal best_len
        if time.perf_counter() > deadline:
            return -1  # timeout
        cur = path[-1]
        extended = False
        for b in order:
            v = cur ^ (1 << b)
            if used[v] or blocked[v]:
                continue
            # Lookahead: v needs a valid successor
            ok = False
            for b2 in order:
                s = v ^ (1 << b2)
                if not (used[s] or blocked[s]):
                    ok = True
                    break
            if not ok:
                continue
            # Extend
            path.append(v)
            used[v] = True
            # cur is now forbidden (it's at index len(path)-2, non-consecutive to future)
            # Block cur's neighbors (except v which is now the last)
            for b3 in range(N):
                nb = cur ^ (1 << b3)
                if nb != v:
                    blocked[nb] = True
            rc = backtrack()
            if rc == -1:
                # timeout: undo and propagate
                for b3 in range(N):
                    nb = cur ^ (1 << b3)
                    if nb != v:
                        blocked[nb] = False
                used[v] = False
                path.pop()
                return -1
            # Undo
            for b3 in range(N):
                nb = cur ^ (1 << b3)
                if nb != v:
                    blocked[nb] = False
            used[v] = False
            path.pop()
            extended = True
            if len(path) + 1 > best_len:
                best_len = len(path) + 1
                best_ref[1] = path[:]
        if len(path) > best_len:
            best_len = len(path)
            best_ref[1] = path[:]
        return 1

    backtrack()
    if len(path) > best_len:
        best_len = len(path)
        best_ref[1] = path[:]
    return best_len


def construct():
    t0 = time.perf_counter()
    deadline = t0 + _DEADLINE

    # Baseline greedy
    gpath = _greedy(list(range(N)))
    best_len = len(gpath)
    best_path = gpath[:]

    rng = random.Random(42)
    best_ref = [best_len, best_path]

    while time.perf_counter() < deadline - 2.0:
        order = list(range(N))
        rng.shuffle(order)
        start = rng.randrange(SIZE)
        _search(order, start, deadline, best_ref)
        if time.perf_counter() > deadline - 2.0:
            break

    result = best_ref[1]
    if result is None or len(result) < 2:
        return gpath
    if not _valid_snake(result):
        return gpath
    return result
