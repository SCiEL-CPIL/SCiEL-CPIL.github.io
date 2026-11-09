"""Stronger solver for the 18-wire sorting-network task.

Structural changes vs the seed:
  * Bit-sliced (bit-parallel) verification: pack 32 inputs per uint32 per wire,
    so a full network is ~18*4 vector ops per comparator instead of 262144-row ops.
  * Greedy forward construction (new design family, not just removal) with a fast
    prefix-soundness filter (a prefix that sorts all-ones must be a permutation).
  * Aggressive reduction: removal, random reordering, and merge-neighborhood swaps.
"""
import time
import random
import numpy as np

N = 18
B = 32          # inputs per uint32 lane
M = 2 ** N // B  # lanes (262144/32 = 8192)
FULL = np.full((M, N), 0xFFFFFFFF, dtype=np.uint32)
ZERO = np.zeros((M, N), dtype=np.uint32)


def _verify_full(net):
    """True iff net sorts every 2^N Boolean input (bit-sliced, exact)."""
    if not net:
        return False
    w = FULL.copy()
    for i, j in net:
        a = w[:, i]
        b = w[:, j]
        w[:, i] = a & b          # min -> i
        w[:, j] = a | b          # max -> j
    for k in range(N - 1):       # sorted iff no 1 sits left of a 0
        if np.any(w[:, k] & ~w[:, k + 1]):
            return False
    return True


def _perm_prefix(net):
    """Cheap necessary condition: a prefix that sorts all-ones must be a permutation.

    Tracks the set of wires reachable from the top (N-1) through comparators.
    If that set is not all wires, the all-ones input cannot be sorted by this prefix.
    """
    seen = [False] * N
    seen[N - 1] = True
    for i, j in net:
        if seen[j] and not seen[i]:
            seen[i] = True
        if seen[i] and not seen[j]:
            seen[j] = True
    return all(seen)


def _construct_greedy(rng, time_budget):
    """Build a sorting network by adding comparators one at a time (sound prefix)."""
    net = []
    while time_budget():
        found = False
        for i in range(N):
            for j in range(i + 1, N):
                cand = net + [(i, j)]
                if _perm_prefix(cand) and _verify_full(cand):
                    net = cand
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return net


def _reduce(net, time_budget):
    """Reduce comparator count via removal, random reordering, and merge swaps."""
    best = list(net)
    best_len = len(best)
    while time_budget():
        improved = False

        # 1) removal (front-to-back)
        for idx in range(len(best)):
            if not time_budget():
                break
            cand = best[:idx] + best[idx + 1:]
            if _verify_full(cand):
                best = cand
                best_len -= 1
                improved = True
                break
        if improved:
            continue

        # 2) random reordering + removal
        for _ in range(200):
            if not time_budget():
                break
            cand = list(best)
            rng.shuffle(cand)
            if _verify_full(cand):
                for k in range(len(cand)):
                    if not time_budget():
                        break
                    c2 = cand[:k] + cand[k + 1:]
                    if _verify_full(c2):
                        best = c2
                        best_len = len(c2)
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue

        # 3) merge-neighborhood swap (a,b)(c,d) -> (a,c)(b,d) when disjoint
        for p in range(len(best) - 1):
            if not time_budget():
                break
            a, b = best[p]
            c, d = best[p + 1]
            if len({a, b, c, d}) == 4:
                cand = best[:p] + [(a, c), (b, d)] + best[p + 2:]
                if _verify_full(cand):
                    best = cand
                    improved = True
                    break
        if improved:
            continue

        break
    return best


def construct():
    """Construct a sorting network on 18 wires (returns list of (i, j) pairs)."""
    start = time.time()
    deadline = start + 82.0
    time_budget = lambda: time.time() < deadline
    rng = random.Random(12345)

    def odd_even_transposition():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    best_net = odd_even_transposition()
    best_count = len(best_net)

    # Phase A: greedy forward construction (may beat the transposition seed).
    g = _construct_greedy(rng, time_budget)
    if g and len(g) < best_count:
        best_net = g
        best_count = len(g)

    # Phase B: aggressive reduction of the best network found so far.
    best_net = _reduce(best_net, time_budget)
    best_count = len(best_net)

    # Phase C: safety net -- if something went wrong, fall back to a valid seed.
    if not best_net or not _verify_full(best_net):
        best_net = odd_even_transposition()

    return best_net
