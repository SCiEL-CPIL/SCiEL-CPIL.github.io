"""Snake-in-the-box in Q_9: time-bounded randomized greedy search (keep greedy family)."""
import time
import random

N = 9
V = 1 << N
SEED = 104.0
DEADLINE = 150.0
SAFETY = 10.0

# Precompute neighbors for every vertex (9 neighbors each).
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(V)]

# Bit masks for bit order permutations
def _bit_order(rng):
    order = list(range(N))
    rng.shuffle(order)
    return order


def _is_valid_snake(path):
    """Verify induced path: consecutive differ by 1 bit, non-consecutive by >=2 bits."""
    if not path:
        return True
    seen = set(path)
    if len(seen) != len(path):
        return False
    n = len(path)
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    # non-consecutive must differ by >= 2 bits
    for i in range(n):
        vi = path[i]
        for j in range(i + 2, n):
            if (vi ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_once(rng, start, bit_order):
    """One greedy extension run. Returns the path list."""
    path = [start]
    used = {start}
    # neighbor set of vertices already in the path (for fast "differs by 1 bit" test)
    neigh_of_path = set(_NEIGHBORS[start])
    while True:
        cur = path[-1]
        best = None
        best_score = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if v in used:
                continue
            # v must differ by >=2 bits from every non-consecutive path vertex,
            # equivalently v must not be a neighbor of any path vertex except cur.
            # cur is a neighbor of v by construction; check others.
            if v in neigh_of_path:
                # v is adjacent to some path vertex; only allowed if that vertex is cur
                # but v is adjacent to cur always, so we need to ensure no OTHER adjacency.
                # Since neigh_of_path includes cur's neighbors and v==cur^bit, v is in it.
                # We must check if v is adjacent to any vertex other than cur.
                conflict = False
                for p in path:
                    if p == cur:
                        continue
                    if v in _NEIGHBORS[p]:
                        conflict = True
                        break
                if conflict:
                    continue
            # candidate is valid; score by number of newly-blocked neighbors (fewer = better)
            # count how many of v's neighbors are already in used
            block = 0
            for nb in _NEIGHBORS[v]:
                if nb in used:
                    block += 1
            if best_score is None or block < best_score:
                best_score = block
                best = v
                if block == 0:
                    break
        if best is None:
            return path
        path.append(best)
        used.add(best)
        neigh_of_path.update(_NEIGHBORS[best])
    return path


def _greedy_once_fast(rng, start, bit_order):
    """Faster variant using a global adjacency-to-path counter."""
    path = [start]
    used = [False] * V
    used[start] = True
    # adj_count[x] = number of path vertices adjacent to x
    adj_count = [0] * V
    for nb in _NEIGHBORS[start]:
        adj_count[nb] += 1
    while True:
        cur = path[-1]
        best = None
        best_score = None
        for b in bit_order:
            v = cur ^ (1 << b)
            if used[v]:
                continue
            # v is adjacent to cur (counted). It is a valid extension iff adj_count[v] == 1
            # (only cur is adjacent among path vertices).
            if adj_count[v] != 1:
                continue
            # score: number of neighbors of v already used
            block = 0
            for nb in _NEIGHBORS[v]:
                if used[nb]:
                    block += 1
            if best_score is None or block < best_score:
                best_score = block
                best = v
                if block == 0:
                    break
        if best is None:
            return path
        path.append(best)
        used[best] = True
        for nb in _NEIGHBORS[best]:
            adj_count[nb] += 1
    return path


def construct():
    rng = random.Random(20260715)
    t0 = time.time()
    best_path = None
    best_len = 0

    # Always run the deterministic seed-like greedy first for a baseline.
    base = _greedy_once_fast(rng, 0, list(range(N)))
    best_path = base
    best_len = len(base) - 1

    deadline = t0 + DEADLINE - SAFETY
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        start = rng.randrange(V)
        bit_order = _bit_order(rng)
        path = _greedy_once_fast(rng, start, bit_order)
        L = len(path) - 1
        if L > best_len:
            best_len = L
            best_path = path
            if best_len >= 190:
                break

    if best_path is None:
        best_path = [0]
    # Final safety: ensure validity of the best path.
    if not _is_valid_snake(best_path):
        # fall back to a verified short path
        for s in range(0, V, 37):
            p = _greedy_once_fast(rng, s, list(range(N)))
            if _is_valid_snake(p) and len(p) - 1 >= best_len:
                best_path = p
                break
        if not _is_valid_snake(best_path):
            best_path = [0, 1]
    return best_path
