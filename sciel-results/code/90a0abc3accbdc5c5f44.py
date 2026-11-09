"""Find maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

_LCM = 232792560
_scaled = [(_LCM // i) for i in range(1, 23)]

def _check(vals):
    """Check if all subset sums of vals are distinct."""
    k = len(vals)
    if k <= 1:
        return True
    sums = {0}
    for v in vals:
        new = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new:
                return False
            new.add(ns)
        sums |= new
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 80

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    blen = len(best)
    elems = list(range(1, 23))

    def greedy(order):
        res = []
        for e in order:
            res.append(e)
            if not _check([_scaled[x - 1] for x in res]):
                res.pop()
        return res

    # Fixed orderings
    for order in [
        sorted(elems, reverse=True),
        sorted(elems, key=lambda x: _scaled[x - 1]),
        sorted(elems, key=lambda x: -_scaled[x - 1]),
        elems,
    ]:
        r = greedy(order)
        if len(r) > blen:
            best, blen = r, len(r)

    # Random restarts
    rng = random.Random(42)
    for _ in range(3000):
        if time.time() > deadline:
            break
        order = elems[:]
        rng.shuffle(order)
        r = greedy(order)
        if len(r) > blen:
            best, blen = r, len(r)

    # Local improvement: try to add one element
    while time.time() < deadline:
        improved = False
        for e in elems:
            if e in best:
                continue
            test = best + [e]
            if _check([_scaled[x - 1] for x in test]):
                best = test
                blen = len(best)
                improved = True
                break
        if not improved:
            break

    # Try random addition orders
    for _ in range(100):
        if time.time() > deadline:
            break
        cands = [x for x in elems if x not in best]
        rng.shuffle(cands)
        for e in cands:
            test = best + [e]
            if _check([_scaled[x - 1] for x in test]):
                best = test
                blen = len(best)
                break

    return best
