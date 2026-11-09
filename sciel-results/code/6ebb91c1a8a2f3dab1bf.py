"""Minimum maximal no-isosceles set via hypergraph MIS + local search."""
import time
import random
import numpy as np

N = 16
DEADLINE = 320.0  # safety margin under 330s wall-clock


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _precompute():
    """Build bitsets for:
       - conflicts[p]: set of q such that (p, q) can coexist in an independent set? No.
       We want: a set S is independent iff no triple in S is isosceles.
       Represent hyperedges (triples). For each point p, store the set of triples containing p.
       For quick 'can I add p to S' we need: no hyperedge fully contained in S ∪ {p}.
       We precompute for each pair (a,b) the set of c such that (a,b,c) is isosceles.
    """
    P = N * N
    pts = [(x, y) for y in range(N) for x in range(N)]
    # pair_conflicts[a*P+b] = bitset of c where (a,b,c) is isosceles
    pair_conf = [0] * (P * P)
    for a in range(P):
        for b in range(a + 1, P):
            ab = _d2(pts[a], pts[b])
            mask = 0
            for c in range(P):
                if c == a or c == b:
                    continue
                if ab == _d2(pts[a], pts[c]) or ab == _d2(pts[b], pts[c]) or _d2(pts[a], pts[c]) == _d2(pts[b], pts[c]):
                    mask |= (1 << c)
            pair_conf[a * P + b] = mask
            pair_conf[b * P + a] = mask
    return pts, pair_conf


def _can_add(pair_conf, chosen, p):
    """True if adding p to chosen keeps it independent (no isosceles triple)."""
    for i in range(len(chosen)):
        a = chosen[i]
        for j in range(i + 1, len(chosen)):
            b = chosen[j]
            if pair_conf[a * N * N + b] & (1 << p):
                return False
    return True


def _greedy_maximal(pts, pair_conf, order):
    """Build a maximal independent set by greedily adding in given order."""
    chosen = []
    for p in order:
        if _can_add(pair_conf, chosen, p):
            chosen.append(p)
    return chosen


def _is_maximal(pts, pair_conf, chosen):
    """Check that no point outside chosen can be added."""
    chosen_set = set(chosen)
    for p in range(N * N):
        if p in chosen_set:
            continue
        if _can_add(pair_conf, chosen, p):
            return False
    return True


def _local_search(pts, pair_conf, chosen, rng, time_limit):
    """Try to shrink a maximal independent set via swap/remove moves.
       A move: remove one point from chosen, then must re-add? No - removing keeps
       independence but breaks maximality. We need to remove a point AND the set must
       remain maximal. Removing a point p: set is still independent. It is maximal iff
       no outside point can be added. So we can remove p if after removal no new point
       becomes addable. Equivalent: p is 'essential' if some outside point q could be
       added only because of p's absence... Actually removing p makes MORE points addable
       (since fewer constraints), so maximality is harder to maintain. So simple removal
       rarely helps. Instead, do 1-for-1 swaps: remove p, add q (outside). Net size same.
       To reduce size we need to remove without adding - only valid if still maximal.
       We'll focus on minimizing via iterative improvement of structure, then try removals.
    """
    chosen = list(chosen)
    best = len(chosen)
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each point: check maximality after removal
        for idx in range(len(chosen)):
            if time.time() - start > time_limit:
                break
            p = chosen[idx]
            trial = chosen[:idx] + chosen[idx + 1:]
            if _is_maximal(pts, pair_conf, trial):
                chosen = trial
                best = len(chosen)
                improved = True
                break
    return chosen


def construct():
    start = time.time()
    rng = random.Random(12345)
    pts, pair_conf = _precompute()
    P = N * N

    # Phase 1: greedy with several random orders, keep smallest maximal set
    best_set = None
    best_size = P + 1
    order = list(range(P))
    # deterministic row-major first
    s = _greedy_maximal(pts, pair_conf, order)
    if _is_maximal(pts, pair_conf, s):
        best_set = s
        best_size = len(s)

    # random orders
    while time.time() - start < DEADLINE - 60:
        rng.shuffle(order)
        s = _greedy_maximal(pts, pair_conf, order)
        if len(s) < best_size and _is_maximal(pts, pair_conf, s):
            best_set = s
            best_size = len(s)

    # Phase 2: local search to shrink
    if best_set is not None and time.time() - start < DEADLINE - 30:
        best_set = _local_search(pts, pair_conf, best_set, rng, DEADLINE - 30 - (time.time() - start))

    # Phase 3: try more greedy + local search combos
    while time.time() - start < DEADLINE - 10:
        rng.shuffle(order)
        s = _greedy_maximal(pts, pair_conf, order)
        if len(s) < best_size + 5 and _is_maximal(pts, pair_conf, s):
            s = _local_search(pts, pair_conf, s, rng, max(1.0, DEADLINE - 10 - (time.time() - start)))
            if len(s) < best_size:
                best_set = s
                best_size = len(s)

    # Fallback: ensure we return something valid
    if best_set is None:
        best_set = _greedy_maximal(pts, pair_conf, list(range(P)))

    return [pts[i] for i in best_set]
