"""Snake-in-the-box Q_9: insertion + extension local search with multi-restart."""
import time
import random

N = 9
V = 1 << N  # 512
MASK = (1 << V) - 1
DEADLINE = 130.0  # safety margin under the 150 s wall-clock kill

# Precompute neighbor bitmask for every vertex: adj[v] has bit u set iff popcount(v^u)==1
_adj = [0] * V
for _v in range(V):
    m = 0
    for _b in range(N):
        m |= 1 << (_v ^ (1 << _b))
    _adj[_v] = m
ADJ = _adj
# Precompute the list of neighbors for each vertex (for extension)
_NEI = [[v ^ (1 << b) for b in range(N)] for v in range(V)]


def _greedy_start(start=0, rng=None):
    """Greedy end-extension snake from `start`. Returns list of vertices."""
    path = [start]
    used = 1 << start
    while True:
        cur = path[-1]
        nxt = None
        cands = [n for n in _NEI[cur] if not (used >> n) & 1 and (ADJ[n] & (used ^ (1 << cur))) == 0]
        if not cands:
            break
        if rng is not None:
            nxt = rng.choice(cands)
        else:
            nxt = min(cands)
        path.append(nxt)
        used |= 1 << nxt
    return path


def _improve(path, rng):
    """Insertion + extension local search. Returns improved path list."""
    used = 0
    for v in path:
        used |= 1 << v
    n = len(path)
    improved = True
    while improved:
        improved = False
        # Pass 1: insertion
        for i in range(n - 1):
            a = path[i]
            b = path[i + 1]
            # neighbors of a not used and not adjacent to anything except a and b
            cand = ADJ[a] & ~used
            cand &= ~(ADJ[b] ^ (1 << a))  # remove b's neighbors other than a
            # must not be adjacent to any path vertex other than a
            # i.e. (ADJ[c] & used) must equal (1<<a)
            # iterate set bits
            m = cand
            while m:
                lsb = m & (-m)
                c = lsb.bit_length() - 1
                m ^= lsb
                if (ADJ[c] & used) == (1 << a):
                    path.insert(i + 1, c)
                    used |= lsb
                    n += 1
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Pass 2: extend ends
        for idx in (0, 1):
            if idx == 0:
                cur = path[0]
                cand = ADJ[cur] & ~used
                cand &= ~(ADJ[path[1]] ^ (1 << cur))
                # not adjacent to any other path vertex
                m = cand
                found = None
                while m:
                    lsb = m & (-m)
                    c = lsb.bit_length() - 1
                    m ^= lsb
                    if (ADJ[c] & used) == (1 << cur):
                        found = c
                        break
                if found is not None:
                    path.insert(0, found)
                    used |= 1 << found
                    n += 1
                    improved = True
            else:
                cur = path[-1]
                cand = ADJ[cur] & ~used
                cand &= ~(ADJ[path[-2]] ^ (1 << cur))
                m = cand
                found = None
                while m:
                    lsb = m & (-m)
                    c = lsb.bit_length() - 1
                    m ^= lsb
                    if (ADJ[c] & used) == (1 << cur):
                        found = c
                        break
                if found is not None:
                    path.append(found)
                    used |= 1 << found
                    n += 1
                    improved = True
            if improved:
                break
    return path


def construct():
    t0 = time.time()
    best = None
    best_len = 0

    # Deterministic baseline first
    p = _greedy_start(0, None)
    p = _improve(p, None)
    best = p
    best_len = len(p)

    rng = random.Random(12345)
    # Multi-restart: different starts + randomized greedy + improve
    for start in range(V):
        if time.time() - t0 > DEADLINE:
            break
        p = _greedy_start(start, rng)
        p = _improve(p, rng)
        if len(p) > best_len:
            best = p
            best_len = len(p)

    return best


if __name__ == "__main__":
    res = construct()
    print("length", len(res))
