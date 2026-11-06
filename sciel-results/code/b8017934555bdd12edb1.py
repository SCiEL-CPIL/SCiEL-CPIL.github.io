"""Greedy B2[2] construction with fast combined validity + randomized restarts."""
import random
import time

N = 1000
MAX_REP = 2


def _greedy(order, counts, chosen):
    """Extend current state by greedily scanning `order`; return new counts, chosen."""
    for x in order:
        ok = True
        # a <= b, a,b in chosen (or a = x). Representations of s = x + y (y in chosen) or 2x.
        for y in chosen:
            s = x + y
            c = counts.get(s, 0)
            # y already contributes one rep (x<=y or x==y handled by counting each pair once).
            # New rep count = c + 1 (for pair (x,y)). Need <= MAX_REP.
            if c + 1 > MAX_REP:
                ok = False
                break
        if ok:
            s = 2 * x
            if counts.get(s, 0) + 1 > MAX_REP:
                ok = False
        if ok:
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s = 2 * x
            counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
    return counts, chosen


def construct():
    deadline = time.time() + 115.0  # safety margin under 120s kill
    rng = random.Random(12345)

    best = None
    best_size = -1

    def consider(chosen):
        nonlocal best, best_size
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = list(chosen)

    # Baseline: deterministic ascending greedy (the seed behavior).
    counts, chosen = _greedy(range(1, N + 1), {}, [])
    consider(chosen)

    # Randomized restarts: random orderings + random subsets of candidates.
    base = list(range(1, N + 1))
    while time.time() < deadline:
        mode = rng.random()
        if mode < 0.5:
            order = base[:]
            rng.shuffle(order)
            counts, chosen = _greedy(order, {}, [])
        elif mode < 0.8:
            # random permutation but bias toward small numbers first (keep a fraction sorted)
            order = base[:]
            k = rng.randint(200, 700)
            order[:k] = sorted(order[:k])
            rng.shuffle(order)
            counts, chosen = _greedy(order, {}, [])
        else:
            # start from a random seed subset, then fill
            seed = rng.sample(base, rng.randint(50, 200))
            counts, chosen = _greedy(seed, {}, [])
            rest = [v for v in base if v not in set(seed)]
            rng.shuffle(rest)
            counts, chosen = _greedy(rest, counts, chosen)
        consider(chosen)

    if best is None:
        best = list(range(1, N + 1))[:1]
    return best
