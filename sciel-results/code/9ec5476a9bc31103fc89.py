import time
import random
import sys

_START = None
_DEADLINE = 110.0  # safety margin under 120s kill; leave room for interpreter overhead

def _elapsed():
    return time.monotonic() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _init_sums(a):
    sums = set()
    n = len(a)
    for i in range(n):
        ai = a[i]
        for j in range(i, n):
            sums.add(ai + a[j])
    return sums

def _can_add(a, sums, x):
    # new sums: x + y for y in a (all), plus 2*x
    # need all distinct and not in sums
    seen = set()
    for y in a:
        s = x + y
        if s in sums or s in seen:
            return False
        seen.add(s)
    s2 = 2 * x
    if s2 in sums or s2 in seen:
        return False
    return True

def _add(a, sums, x):
    for y in a:
        sums.add(x + y)
    sums.add(2 * x)
    a.append(x)
    a.sort()

def _remove(a, sums, x):
    # rebuild sums is O(n^2); n small (~20), acceptable
    a.remove(x)
    return _init_sums(a)

def _greedy_fill(a, sums, candidates, rng, bias='high'):
    # try to add as many as possible from candidates
    # candidates: list of ints
    changed = True
    while changed and _time_left() > 1.0:
        changed = False
        # order candidates
        if bias == 'low':
            order = list(candidates)
            order.sort()
        elif bias == 'high':
            order = list(candidates)
            order.sort(reverse=True)
        else:  # random
            order = list(candidates)
            rng.shuffle(order)
        for x in order:
            if x in a:
                continue
            if _can_add(a, sums, x):
                _add(a, sums, x)
                candidates.remove(x)
                changed = True
                if _time_left() < 1.0:
                    break
        # if we added, recompute remaining candidates
        if changed:
            candidates = [c for c in range(1, 285) if c not in a]
    return a, sums, candidates

def _local_search(a, sums, rng, max_iters=2000):
    # hill climbing: try swaps to increase size
    best_a = list(a)
    best_sums = set(sums)
    best_size = len(a)
    cur_a = list(a)
    cur_sums = set(sums)
    iters = 0
    while iters < max_iters and _time_left() > 2.0:
        iters += 1
        improved = False
        # try to add a new element (increase size)
        cand = [c for c in range(1, 285) if c not in cur_a]
        rng.shuffle(cand)
        for x in cand:
            if _can_add(cur_a, cur_sums, x):
                _add(cur_a, cur_sums, x)
                improved = True
                break
        if improved:
            if len(cur_a) > best_size:
                best_size = len(cur_a)
                best_a = list(cur_a)
                best_sums = set(cur_sums)
            continue
        # try swap: remove r, add x
        # pick a random element to remove
        if len(cur_a) >= 2:
            r = rng.choice(cur_a)
            # candidate adds
            cand2 = [c for c in range(1, 285) if c not in cur_a]
            rng.shuffle(cand2)
            # limit candidates tried for speed
            for x in cand2[:80]:
                # simulate
                trial = [v for v in cur_a if v != r]
                ts = _init_sums(trial)
                if _can_add(trial, ts, x):
                    # accept if size not worse (same size, different set) - accept to explore
                    cur_a = trial + [x]
                    cur_a.sort()
                    cur_sums = ts
                    for y in cur_a:
                        cur_sums.add(x + y)
                    cur_sums.add(2 * x)
                    if len(cur_a) > best_size:
                        best_size = len(cur_a)
                        best_a = list(cur_a)
                        best_sums = set(cur_sums)
                    improved = True
                    break
        if not improved:
            # stuck; small kick
            if len(cur_a) >= 3:
                # remove 1 random, keep rest
                r = rng.choice(cur_a)
                cur_a.remove(r)
                cur_sums = _init_sums(cur_a)
    return best_a, best_sums, best_size

def construct():
    global _START
    _START = time.monotonic()
    rng = random.Random(12345)
    base = [1, 2, 5, 11]
    best_size = len(base)
    best_a = list(base)
    best_sums = _init_sums(base)

    # Multi-start randomized greedy + local search
    start = 0
    while _time_left() > 3.0:
        start += 1
        a = list(base)
        sums = _init_sums(a)
        # randomized greedy fill with random order, maybe a few passes
        candidates = [c for c in range(1, 285) if c not in a]
        # do a random greedy
        _greedy_fill(a, sums, candidates, rng, bias='random')
        # local search
        a, sums, sz = _local_search(a, sums, rng, max_iters=300)
        if sz > best_size:
            best_size = sz
            best_a = list(a)
            best_sums = set(sums)
            if best_size >= 20:
                break
        # kick variant: start from best and perturb
        if start % 3 == 0 and _time_left() > 3.0:
            a = list(best_a)
            sums = _init_sums(a)
            # remove a couple random non-base elements
            removable = [v for v in a if v not in base]
            k = min(2, len(removable))
            for _ in range(k):
                if removable:
                    r = rng.choice(removable)
                    a.remove(r)
                    removable.remove(r)
            sums = _init_sums(a)
            candidates = [c for c in range(1, 285) if c not in a]
            _greedy_fill(a, sums, candidates, rng, bias='random')
            a, sums, sz = _local_search(a, sums, rng, max_iters=300)
            if sz > best_size:
                best_size = sz
                best_a = list(a)
                best_sums = set(sums)
                if best_size >= 20:
                    break

    return best_a
