"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
# For each 4-set, list its 4 triples (combinations of 3 of its 4 points)
_4SET_TRIPLES = {}
for _f in combinations(range(V), 4):
    _4SET_TRIPLES[frozenset(_f)] = list(combinations(_f, 3))


def _check_triple(trips, a, b, c):
    """Return True if adding {a,b,c} keeps the system K(2,2,2)-free."""
    # For each pair of points (x,y) not in {a,b,c}, check whether the 4-set
    # {a,b,x,y} has all 4 of its triples present (which would complete a
    # K(2,2,2) with the new triple).
    for x in range(V):
        if x == a or x == b or x == c:
            continue
        for y in range(x + 1, V):
            if y == a or y == b or y == c:
                continue
            if ((a, b, x) in trips or (a, b, y) in trips
                    or (a, x, y) in trips or (b, x, y) in trips):
                return False
    # Symmetric for pairs involving c
    for x in range(V):
        if x == a or x == b or x == c:
            continue
        for y in range(x + 1, V):
            if y == a or y == b or y == c:
                continue
            if ((a, c, x) in trips or (a, c, y) in trips
                    or (a, x, y) in trips or (c, x, y) in trips):
                return False
    for x in range(V):
        if x == a or x == b or x == c:
            continue
        for y in range(x + 1, V):
            if y == a or y == b or y == c:
                continue
            if ((b, c, x) in trips or (b, c, y) in trips
                    or (b, x, y) in trips or (c, x, y) in trips):
                return False
    return True


def _greedy(order, deadline):
    trips = set()
    for t in order:
        if time.time() >= deadline:
            break
        a, b, c = t
        if _check_triple(trips, a, b, c):
            trips.add(t)
    return trips


def _repair(trips, deadline):
    """Try to add back triples that were blocked; repeat until stable or deadline."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for t in ALL_TRIPLES:
            if t in trips:
                continue
            a, b, c = t
            if _check_triple(trips, a, b, c):
                trips.add(t)
                changed = True
        if time.time() >= deadline:
            break
    return trips


def _local_search(trips, deadline):
    """Try swapping: remove one triple, then repair to see if we gain."""
    best = trips
    for t in list(best):
        if time.time() >= deadline:
            break
        cand = set(best)
        cand.discard(t)
        cand = _repair(cand, deadline)
        if len(cand) > len(best):
            best = cand
    return best


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # Seed baseline: all triples through point 0 (105 triples, K(2,2,2)-free)
    best = set((0, a, b) for a in range(1, V) for b in range(a + 1, V))

    # Multiple randomized greedy starts
    rng = random.Random(12345)
    attempt = 0
    while time.time() < deadline - 5.0:
        attempt += 1
        order = list(ALL_TRIPLES)
        rng.shuffle(order)
        # Bias: interleave to avoid early saturation
        trips = _greedy(order, deadline)
        # Repair pass to recover blocked triples
        trips = _repair(trips, deadline)
        if len(trips) > len(best):
            best = trips
        if time.time() >= deadline - 5.0:
            break

    # Local search on best
    if time.time() < deadline - 2.0:
        best = _local_search(best, deadline)

    # Final repair to fill any gaps
    best = _repair(best, deadline)

    return [list(t) for t in sorted(best)]


if __name__ == "__main__":
    res = construct()
    print(len(res))
