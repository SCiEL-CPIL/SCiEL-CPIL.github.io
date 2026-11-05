import random
import time

N = 16
DEADLINE = 135.0  # safety margin under 150s wall-clock
RNG = random.Random(20240517)


def _valid(triples):
    """Check pairwise at most one shared point and points in range."""
    for t in triples:
        if len(set(t)) != 3:
            return False
        for p in t:
            if p < 0 or p >= N:
                return False
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            if len(set(triples[i]) & set(triples[j])) > 1:
                return False
    return True


def _calc_alpha(triples, time_limit=0.4):
    """Exact independence number via branch-and-bound with greedy upper bound."""
    tset = [set(t) for t in triples]
    n = len(tset)

    # Precompute for each point the list of triples it belongs to (to remove)
    # We search over subsets S of points that contain no triple.
    # Maximize |S| such that no triple is subset of S.
    start = time.time()
    best = 0
    best_set = frozenset()

    # Greedy upper bound: include a point if it doesn't complete a triple with
    # already-chosen points; gives a feasible lower bound quickly.
    def greedy_lb(available):
        s = set()
        for p in available:
            cand = s | {p}
            ok = True
            for t in tset:
                if t <= cand:
                    ok = False
                    break
            if ok:
                s = cand
        return s

    # Branch-and-bound over points 0..15 (include/exclude)
    # Upper bound: current_size + remaining_points
    import sys
    sys.setrecursionlimit(10000)

    pts = list(range(N))

    def bound(size, idx):
        # remaining points from idx..N-1
        return size + (N - idx)

    def rec(size, idx, chosen):
        nonlocal best, best_set
        if time.time() - start > time_limit:
            return
        if idx == N:
            if size > best:
                best = size
                best_set = frozenset(chosen)
            return
        # Upper bound prune
        if bound(size, idx) <= best:
            return
        p = pts[idx]
        # Try include p
        ok = True
        for t in tset:
            if t <= (chosen | {p}):
                ok = False
                break
        if ok:
            chosen.add(p)
            rec(size + 1, idx + 1, chosen)
            chosen.discard(p)
        # Exclude p
        rec(size, idx + 1, chosen)

    # Get a strong initial lower bound to tighten pruning
    init = greedy_lb(set(range(N)))
    best = len(init)
    best_set = frozenset(init)

    rec(0, 0, set())
    return best


def _can_add(triples, t):
    for x in triples:
        if len(set(x) & set(t)) > 1:
            return False
    return True


def _greedy_start():
    """Build a small sparse initial packing."""
    triples = []
    used_pairs = set()
    # place a handful of disjoint-ish triples
    pts = list(range(N))
    RNG.shuffle(pts)
    i = 0
    while i + 2 < N and len(triples) < 5:
        t = [pts[i], pts[i+1], pts[i+2]]
        triples.append(t)
        i += 3
    return triples


def construct():
    global DEADLINE
    t0 = time.time()
    deadline = t0 + DEADLINE

    triples = _greedy_start()
    cur_alpha = _calc_alpha(triples)
    best_alpha = cur_alpha
    best_triples = [list(t) for t in triples]

    # Simulated annealing
    temp = 3.0
    cool = 0.9999
    step = 0
    while time.time() < deadline - 1.0:
        step += 1
        # choose a move
        r = RNG.random()
        new_triples = triples
        if r < 0.45:
            # add a random triple
            a, b, c = RNG.sample(range(N), 3)
            cand = [a, b, c]
            if _can_add(triples, cand):
                new_triples = triples + [cand]
            else:
                continue
        elif r < 0.75:
            # remove a random triple
            if len(triples) > 1:
                idx = RNG.randrange(len(triples))
                new_triples = triples[:idx] + triples[idx+1:]
            else:
                continue
        else:
            # swap: remove one, add another
            if len(triples) < 1:
                continue
            idx = RNG.randrange(len(triples))
            rest = triples[:idx] + triples[idx+1:]
            a, b, c = RNG.sample(range(N), 3)
            cand = [a, b, c]
            if _can_add(rest, cand):
                new_triples = rest + [cand]
            else:
                continue

        new_alpha = _calc_alpha(new_triples)
        delta = new_alpha - cur_alpha  # want lower
        if delta <= 0 or RNG.random() < pow(2.718281828, -delta / max(temp, 1e-6)):
            triples = new_triples
            cur_alpha = new_alpha
            if cur_alpha < best_alpha:
                best_alpha = cur_alpha
                best_triples = [list(t) for t in triples]
        temp *= cool
        if temp < 0.05:
            temp = 0.05

    # final validity check
    if not _valid(best_triples):
        best_triples = _greedy_start()
    return best_triples
