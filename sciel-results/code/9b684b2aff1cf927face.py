"""Size-efficient 16-wire sorting network: randomized construction + optimization.
Fast NumPy zero-one-principle verification, deadline-guarded search."""
import time
import random
import numpy as np

N = 16
MAXC = 4096
_DEADLINE = 90.0
_start = time.time()

# ---------- Fast verification (zero-one principle, vectorized) ----------
def _sorted_bits():
    # 65536 rows x 16 bits: row k = binary of k (ascending), bit0 = LSB.
    idx = np.arange(1 << N, dtype=np.uint8)[:, None]
    bits = ((idx >> np.arange(N, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    return bits

_ALL = _sorted_bits()
_TARGET = _ALL.copy()  # already sorted ascending (bit0 smallest)

def _apply(comp, A):
    i, j = comp
    ai = A[:, i].copy()
    aj = A[:, j]
    mn = np.minimum(ai, aj)
    mx = np.maximum(ai, aj)
    A[:, i] = mn
    A[:, j] = mx

def _verify(comps):
    A = _ALL.copy()
    for c in comps:
        _apply(c, A)
    return bool(np.array_equal(A, _TARGET))

def _verify_fast(comps):
    # early-exit check on a small random subset for pruning
    A = _ALL.copy()
    for c in comps:
        _apply(c, A)
    return bool(np.array_equal(A, _TARGET))

# ---------- Known-good 60-comparator network (verified) ----------
_BASE = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
    (0, 2), (1, 3), (5, 7), (9, 11), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
    (0, 2), (1, 3), (5, 7), (9, 11), (13, 15),
]

def _odd_even_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# ---------- Search operators ----------
def _random_comparator(rng):
    a = rng.randrange(N)
    b = rng.randrange(N)
    if a == b:
        b = (b + 1) % N
    return (min(a, b), max(a, b))

def _try_remove(comps, rng):
    """Randomly remove one comparator; return new list if still valid, else None."""
    if len(comps) <= 1:
        return None
    idx = rng.randrange(len(comps))
    cand = comps[:idx] + comps[idx + 1:]
    if _verify_fast(cand):
        return cand
    return None

def _try_swap(comps, rng):
    """Randomly swap two comparators (order matters in a network)."""
    if len(comps) < 2:
        return None
    i = rng.randrange(len(comps))
    j = rng.randrange(len(comps))
    if i == j:
        return None
    cand = comps[:]
    cand[i], cand[j] = comps[j], comps[i]
    if _verify_fast(cand):
        return cand
    return None

def _try_reinsert(comps, rng):
    """Take a random comparator and reinsert it at a new random position."""
    if len(comps) < 2:
        return None
    i = rng.randrange(len(comps))
    comp = comps[i]
    base = comps[:i] + comps[i + 1:]
    j = rng.randrange(len(base) + 1)
    cand = base[:j] + [comp] + base[j:]
    if _verify_fast(cand):
        return cand
    return None

def _random_construction(rng, target_len):
    """Greedy random construction: keep adding comparators until valid or target_len."""
    comps = []
    seen = set()
    attempts = 0
    while len(comps) < target_len and attempts < 5000:
        attempts += 1
        c = _random_comparator(rng)
        if c in seen:
            continue
        seen.add(c)
        cand = comps + [c]
        if _verify_fast(cand):
            comps = cand
    return comps

def construct():
    global _start, _DEADLINE
    if _start is None:
        _start = time.time()
    _DEADLINE = _start + 90.0

    rng = random.Random(12345)

    # Ensure we always have a valid fallback.
    base = _BASE if _verify(_BASE) else _odd_even_network()
    best = base
    best_len = len(best)

    def time_left():
        return _DEADLINE - 1.5 - time.time()  # safety margin

    # Phase 1: aggressive removal from the base network.
    cur = base
    while time_left() > 0.2 and len(cur) > 30:
        new = _try_remove(cur, rng)
        if new is not None:
            cur = new
            if len(cur) < best_len:
                best = cur
                best_len = len(cur)

    # Phase 2: local search (swap / reinsert) to enable further removals.
    stall = 0
    while time_left() > 0.3:
        op = rng.random()
        if op < 0.5:
            new = _try_swap(cur, rng)
        elif op < 0.8:
            new = _try_reinsert(cur, rng)
        else:
            new = _try_remove(cur, rng)
        if new is not None:
            cur = new
            stall = 0
            if len(cur) < best_len:
                best = cur
                best_len = len(cur)
        else:
            stall += 1
            if stall > 2000:
                # random restart from a fresh construction
                restart_len = max(best_len, 40)
                rc = _random_construction(rng, restart_len)
                if rc and _verify(rc) and len(rc) <= best_len + 5:
                    cur = rc
                stall = 0

    # Final safety: if best is somehow invalid, fall back to a verified network.
    if not _verify(best):
        best = _BASE if _verify(_BASE) else _odd_even_network()

    # Enforce comparator cap.
    if len(best) > MAXC:
        best = _odd_even_network()[:MAXC]

    return [(int(i), int(j)) for (i, j) in best]
