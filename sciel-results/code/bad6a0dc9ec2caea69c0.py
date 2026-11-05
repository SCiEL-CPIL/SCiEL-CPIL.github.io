"""Search-based design: find a 12-state binary automaton with a large
shortest reset word, verified by BFS on the subset automaton."""
import time
import random

N = 12
FULL = (1 << N) - 1  # bitmask of all 12 states

# EVOLVE-BLOCK-START
def _eval(a, b):
    """Return (threshold, is_sync). threshold = shortest reset word length,
    or -1 if not synchronizing."""
    trans = [a, b]
    dist = {FULL: 0}
    prev = {FULL: None}
    frontier = [FULL]
    d = 0
    while frontier:
        nxt = []
        for s in frontier:
            for ci in range(2):
                t = trans[ci]
                ns = 0
                for i in range(N):
                    if (s >> i) & 1:
                        ns |= 1 << t[i]
                if ns not in dist:
                    dist[ns] = d + 1
                    prev[ns] = (s, ci)
                    if ns & (ns - 1) == 0:  # singleton -> synchronized
                        return d + 1, True
                    nxt.append(ns)
        frontier = nxt
        d += 1
    return -1, False


def _search(seconds, rng):
    """Random search for a high-threshold synchronizing automaton."""
    best = -1
    best_a = None
    best_b = None
    deadline = time.time() + seconds
    while time.time() < deadline:
        # letter a: a cycle plus a few random perturbations
        a = list(range(1, N)) + [0]
        for _ in range(3):
            i = rng.randrange(N)
            a[i] = rng.randrange(N)
        # letter b: random
        b = [rng.randrange(N) for _ in range(N)]
        th, ok = _eval(a, b)
        if ok and th > best:
            best = th
            best_a = a[:]
            best_b = b[:]
    return best, best_a, best_b


def construct():
    """Return two length-N destination lists defining a synchronizing
    12-state binary automaton with a (searched-for) large threshold."""
    t0 = time.time()
    total = 75.0  # safety margin under the 90s kill
    # deterministic fallback (always valid, threshold 12)
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N
    best, best_a, best_b = _search(total, random.Random(20240517))
    if best < 0 or best_a is None:
        return fallback_a, fallback_b
    return best_a, best_b
# EVOLVE-BLOCK-END
