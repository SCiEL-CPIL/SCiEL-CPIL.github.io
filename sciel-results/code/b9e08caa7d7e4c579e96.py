"""Stronger snake search for Q_9: vectorized feasibility + randomized beam + local repair."""
import time
import random

N = 9
SIZE = 1 << N            # 512 vertices
MASK = SIZE - 1

# Precompute the 9 one-bit neighbours for every vertex.
NEIGH = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]

# Precompute a bit_count table for 16-bit chunks (fast popcount via two lookups).
_BC16 = [0] * 65536
for _i in range(65536):
    _BC16[_i] = _i.bit_count()


def _popcount(x):
    return _BC16[x & 0xFFFF] + _BC16[(x >> 16) & 0xFFFF]


def _is_valid_snake(path):
    """Return True if path is an induced path (valid snake) in Q_N."""
    n = len(path)
    if n == 0:
        return True
    seen = [0] * SIZE
    for v in path:
        if not (0 <= v < SIZE):
            return False
        if seen[v]:
            return False
        seen[v] = 1
    for i in range(n - 1):
        if _popcount(path[i] ^ path[i + 1]) != 1:
            return False
    for i in range(n):
        a = path[i]
        for b in range(N):
            nb = a ^ (1 << b)
            for j in range(n):
                if i == j or j == i + 1 or j == i - 1:
                    continue
                if path[j] == nb:
                    return False
    return True


class Snake:
    __slots__ = ("path", "used", "forb", "cur")

    def __init__(self, path):
        self.path = path
        self.used = bytearray(SIZE)
        self.forb = bytearray(SIZE)
        for v in path:
            self.used[v] = 1
        self.cur = path[-1]
        for v in path:
            for nb in NEIGH[v]:
                if nb != path[0] and nb != path[-1]:
                    self.forb[nb] = 1


def _extend(s, v):
    """Return a new Snake extended by v, or None if v is not a valid next vertex."""
    if s.used[v] or s.forb[v]:
        return None
    p = s.path
    new_used = bytearray(s.used)
    new_forb = bytearray(s.forb)
    new_used[v] = 1
    for nb in NEIGH[v]:
        if nb != s.cur and nb != v:
            new_forb[nb] = 1
    return Snake.__new__(Snake), None


def _extend_fast(s, v):
    """In-place-ish extension returning a new Snake (allocates copies)."""
    if s.used[v] or s.forb[v]:
        return None
    new_used = bytearray(s.used)
    new_forb = bytearray(s.forb)
    new_used[v] = 1
    cur = s.cur
    for nb in NEIGH[v]:
        if nb != cur:
            new_forb[nb] = 1
    ns = Snake.__new__(Snake)
    ns.path = s.path + (v,)
    ns.used = new_used
    ns.forb = new_forb
    ns.cur = v
    return ns


def _greedy(seed=0, rng=random, max_len=None):
    """Randomized greedy: at each step pick a random valid extension (with a slight
    preference for lower-degree options to avoid dead-ends)."""
    s = Snake([seed])
    while True:
        cur = s.cur
        cands = [nb for nb in NEIGH[cur] if not s.used[nb] and not s.forb[nb]]
        if not cands:
            break
        if len(cands) == 1:
            v = cands[0]
        else:
            # Prefer candidates whose own non-forbidden degree is small (heuristic),
            # but with randomness so repeated calls differ.
            scored = []
            for c in cands:
                deg = 0
                for nb2 in NEIGH[c]:
                    if not s.used[nb2] and not s.forb[nb2] and nb2 != cur:
                        deg += 1
                scored.append((deg, rng.random(), c))
            scored.sort()
            # pick among the best few with randomness
            k = min(len(scored), 3)
            _, _, v = scored[rng.randrange(k)]
        s = _extend_fast(s, v)
        if max_len and len(s.path) >= max_len:
            break
    return list(s.path)


def _beam_search(rng, beam_width, max_iter, deadline, best_holder):
    """Beam search from a random seed. best_holder is a list; we update it if we beat it."""
    seed = rng.randrange(SIZE)
    beam = [Snake([seed])]
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        new_beam = []
        seen_next = set()
        for s in beam:
            cur = s.cur
            cands = [nb for nb in NEIGH[cur] if not s.used[nb] and not s.forb[nb]]
            if not cands:
                continue
            # keep the top few by (small degree, random) to control branching
            scored = []
            for c in cands:
                deg = 0
                for nb2 in NEIGH[c]:
                    if not s.used[nb2] and not s.forb[nb2] and nb2 != cur:
                        deg += 1
                scored.append((deg, rng.random(), c))
            scored.sort()
            take = min(len(scored), 2)
            for _, _, c in scored[:take]:
                ns = _extend_fast(s, c)
                if ns is None:
                    continue
                key = (ns.cur, len(ns.path))
                if key in seen_next:
                    continue
                seen_next.add(key)
                new_beam.append(ns)
        if not new_beam:
            break
        # prune to beam_width by length (longest first), tie-break random
        new_beam.sort(key=lambda s: (-len(s.path), rng.random()))
        beam = new_beam[:beam_width]
        bl = max(len(s.path) for s in beam)
        if bl > best_holder[0]:
            best_holder[0] = bl
            best_holder[1] = list(beam[0].path)
    # record any beam member
    for s in beam:
        if len(s.path) > best_holder[0]:
            best_holder[0] = len(s.path)
            best_holder[1] = list(s.path)


def _local_repair(path, rng, deadline):
    """Try to extend a snake from its end by backtracking a few steps and trying
    alternative branches. Returns a longer path or the original."""
    best = list(path)
    base = Snake(best)
    # Attempt a bounded depth-first extension with random branch order.
    stack = [(base, 0)]
    while stack and time.time() < deadline:
        s, depth = stack.pop()
        if len(s.path) > len(best):
            best = list(s.path)
        if depth >= 4:
            continue
        cur = s.cur
        cands = [nb for nb in NEIGH[cur] if not s.used[nb] and not s.forb[nb]]
        rng.shuffle(cands)
        for c in cands:
            ns = _extend_fast(s, c)
            if ns is not None:
                stack.append((ns, depth + 1))
    return best


def construct():
    start = time.time()
    deadline = start + 145.0   # safety margin before the 150 s kill

    best = []
    best_len = 0
    best_holder = [best_len, best]

    # 1) A few fast randomized greedy runs to establish a baseline quickly.
    for seed in range(0, SIZE, 16):
        if time.time() > deadline - 30.0:
            break
        p = _greedy(seed, random)
        if len(p) > best_len:
            best_len = len(p)
            best = p

    # 2) Beam search with randomized seeds and widths.
    rng = random.Random(12345)
    for _ in range(200):
        if time.time() > deadline - 10.0:
            break
        bw = rng.choice([8, 16, 32, 64])
        mi = rng.choice([200, 400, 800])
        _beam_search(rng, bw, mi, deadline, best_holder)
        best_len = best_holder[0]
        best = best_holder[1]

    # 3) Local repair on the best snake found so far.
    if time.time() < deadline - 5.0:
        repaired = _local_repair(best, rng, deadline)
        if len(repaired) > best_len:
            best = repaired
            best_len = len(repaired)

    # 4) Final safety: verify and return.
    if not _is_valid_snake(best):
        # Fallback: a guaranteed valid snake (the greedy one from seed 0).
        best = _greedy(0, random)

    return best
