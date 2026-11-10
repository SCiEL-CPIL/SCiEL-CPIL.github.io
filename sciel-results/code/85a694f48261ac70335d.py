"""Rewritten program: bitset-accelerated greedy + local-search polish."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAXC = 48
DEADLINE = 55.0          # hard wall-clock guard (evaluator kills at 90s)
POLISH_BUDGET = 40.0     # time for the local-search phase

# ---------- bitset precomputation (done once) ----------
# bad[a] is a Python int whose bit x is 1 iff x divides some sum of 3 distinct
# OTHER selected numbers that include a and two others.  When building a set S,
# x is safe w.r.t. a in S  <=>  (bad[a] >> x) & 1 == 0.
_bad = [0] * (N + 1)
for a in range(1, N + 1):
    m = 0
    for u in range(1, N + 1):
        if u == a:
            continue
        for v in range(u + 1, N + 1):
            if v == a:
                continue
            if (a + u + v) % a == 0:
                m |= (1 << u) | (1 << v)
    _bad[a] = m


def _can_add(x, S):
    """True if adding x keeps every a in S from dividing a triple sum that includes x."""
    for a in S:
        if (_bad[a] >> x) & 1:
            return False
    return True


def _greedy(order):
    S = []
    for x in order:
        if _can_add(x, S):
            S.append(x)
    return S


def _local_search(S, deadline, rng):
    """Try single swaps (remove r, add c) to increase cardinality."""
    best = list(S)
    best_len = len(best)
    while time.time() < deadline:
        improved = False
        cands = [x for x in range(1, N + 1) if x not in best]
        rng.shuffle(cands)
        for c in cands:
            for r in best:
                T = [y for y in best if y != r]
                if _can_add(c, T):
                    T.append(c)
                    if len(T) > best_len:
                        best = T
                        best_len = len(best)
                        improved = True
                        cands = [x for x in range(1, N + 1) if x not in best]
                        rng.shuffle(cands)
        if not improved:
            break
    return best


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Phase 1: build strong greedy seeds from several orderings.
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(2, N + 1, 2)),
        list(range(1, N + 1, 2)),
        list(range(N, 0, -2)),
        list(range(1, N + 1, 3)),
    ]
    best = []
    for o in orders:
        if time.time() - t0 > 5.0:
            break
        S = _greedy(o)
        if len(S) > len(best):
            best = S

    # Phase 2: local-search polish within the time budget.
    if time.time() - t0 < DEADLINE - POLISH_BUDGET - 2.0:
        best = _local_search(best, min(DEADLINE - 2.0, t0 + POLISH_BUDGET), rng)

    return best
# EVOLVE-BLOCK-END
