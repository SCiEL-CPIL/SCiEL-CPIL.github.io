"""Maximum Independent Set priority scorer.

Structural change vs the seed: instead of a single static "lowest degree"
score, we *simulate* several dynamic greedy orderings offline, measure the
independent-set size each one actually produces, and emit the ordering that
wins as strictly-decreasing scores.  The fixed greedy decoder then reproduces
that exact ordering (higher score = picked earlier), so any better greedy
strategy we can run here is transferred to the decoder for free.
"""
# EVOLVE-BLOCK-START
import time
import random

import numpy as np

_DEADLINE_S = 110.0          # hard wall-clock guard (evaluator kills at ~120s)
_MARGIN_S = 8.0              # safety margin before the deadline


def _build_neigh(adjacency, n):
    """Neighbor lists + original degrees from a bool adjacency matrix."""
    neigh = [list(np.nonzero(adjacency[i])[0]) for i in range(n)]
    orig_deg = [len(neigh[i]) for i in range(n)]
    return neigh, orig_deg


def _run_greedy(neigh, n, orig_deg, mode, rng):
    """Run one greedy MIS ordering; return (order, size).

    `remaining` always holds only vertices not adjacent to any already-picked
    vertex, so `order` is a valid independent set and its length is the size.
    """
    remaining = set(range(n))
    cur_deg = [len(neigh[i]) for i in range(n)]
    order = []

    while remaining:
        if mode == "min_deg":
            m = min(cur_deg[v] for v in remaining)
            cands = [v for v in remaining if cur_deg[v] == m]
        elif mode == "min_static":
            m = min(orig_deg[v] for v in remaining)
            cands = [v for v in remaining if orig_deg[v] == m]
        else:  # "min_sum": minimize sum of current degrees of live neighbors
            def _sc(v):
                s = 0
                for u in neigh[v]:
                    if u in remaining:
                        s += cur_deg[u]
                return s
            m = min(_sc(v) for v in remaining)
            cands = [v for v in remaining if _sc(v) == m]

        v = rng.choice(cands)
        order.append(v)

        # Remove v and every still-live neighbor of v.
        to_remove = [v] + [u for u in neigh[v] if u in remaining]
        for u in to_remove:
            for w in neigh[u]:
                if w in remaining:
                    cur_deg[w] -= 1
        for u in to_remove:
            remaining.discard(u)

    return order, len(order)


def priority(adjacency):
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    neigh, orig_deg = _build_neigh(adjacency, n)

    best_order = None
    best_size = -1
    best_mode = "min_static"

    def _consider(order, size, mode, seed):
        nonlocal best_order, best_size, best_mode
        if size > best_size:
            best_size = size
            best_order = order
            best_mode = mode

    # Deterministic seed-level baseline + a couple of cheap dynamic starts.
    for mode, seed in (("min_static", 0), ("min_deg", 0), ("min_sum", 0)):
        _consider(*_run_greedy(neigh, n, orig_deg, mode, random.Random(seed)),
                  mode, seed)

    # Randomized restarts of the dynamic heuristics; keep the best ordering.
    # min_deg is cheap, min_sum is heavier, so weight them accordingly.
    mode_pool = (["min_deg"] * 3) + (["min_sum"] * 1) + (["min_deg"] * 1)
    t0 = time.time()
    limit = _DEADLINE_S - _MARGIN_S
    seed = 1
    while (time.time() - t0) < limit:
        mode = random.choice(mode_pool)
        order, size = _run_greedy(neigh, n, orig_deg, mode, random.Random(seed))
        _consider(order, size, mode, seed)
        seed += 1

    if best_order is None:
        best_order = list(range(n))

    # Strictly decreasing scores in pick order => decoder reproduces the order
    # with no tie ambiguity.
    scores = np.empty(n, dtype=float)
    for idx, v in enumerate(best_order):
        scores[v] = float(n - idx)

    return scores
# EVOLVE-BLOCK-END
