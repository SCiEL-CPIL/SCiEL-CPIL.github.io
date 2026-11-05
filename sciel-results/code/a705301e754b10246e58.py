"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START = None
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _forbidden_triples(n=180):
    """Return list of frozensets {a,b,c} with a<b<c and 1/a=1/b+1/c."""
    triples = []
    # a is smallest: 1/a = 1/b + 1/c => b*c = a*(b+c) => (b-a)(c-a)=a^2
    for a in range(1, n + 1):
        a2 = a * a
        # u*v = a^2, u = b-a, v = c-a, with 0<u<v (so b<c) and b,c <= n
        # enumerate divisors u of a^2
        u = 1
        while u * u <= a2:
            if a2 % u == 0:
                v = a2 // u
                # pair (u, v) with u < v
                if u < v:
                    b = a + u
                    c = a + v
                    if b <= n and c <= n and b != c:
                        triples.append((a, b, c))
                # also consider (v, u) but that gives b>c, skip since we want a<b<c
            u += 1
    return triples

def _conflict_pairs(triples):
    """For each element, the set of other elements that appear in a forbidden triple with it.
    Actually we need: for a candidate set, no triple fully contained.
    We'll precompute for each pair (x,y) the list of z such that {x,y,z} is forbidden.
    """
    from collections import defaultdict
    pair_to_third = defaultdict(list)
    for (a, b, c) in triples:
        pair_to_third[(a, b)].append(c)
        pair_to_third[(b, a)].append(c)
        pair_to_third[(a, c)].append(b)
        pair_to_third[(c, a)].append(b)
        pair_to_third[(b, c)].append(a)
        pair_to_third[(c, b)].append(a)
    return pair_to_third

def _is_valid_set(s, pair_to_third):
    """Check if set s has no forbidden triple."""
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            x, y = lst[i], lst[j]
            for z in pair_to_third.get((x, y), []):
                if z in s and z != x and z != y:
                    return False
    return True

def _greedy_construct(order, pair_to_third, n=180):
    """Greedy: add elements in given order if valid."""
    s = set()
    for x in order:
        # check if adding x creates a forbidden triple
        ok = True
        for y in s:
            for z in pair_to_third.get((x, y), []):
                if z in s and z != x and z != y:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            s.add(x)
    return s

def _local_search(s, pair_to_third, n=180, max_iters=200000, deadline_check=None):
    """Try to improve s by adding, removing, or swapping elements."""
    s = set(s)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        if deadline_check and deadline_check():
            break
        improved = False
        iters += 1
        # Try adding a missing element
        missing = [x for x in range(1, n + 1) if x not in s]
        random.shuffle(missing)
        added = False
        for x in missing:
            ok = True
            for y in s:
                for z in pair_to_third.get((x, y), []):
                    if z in s and z != x and z != y:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                s.add(x)
                improved = True
                added = True
                break
        if added:
            continue
        # Try swapping: remove one, add another
        lst = list(s)
        random.shuffle(lst)
        swapped = False
        for rem in lst:
            s2 = s - {rem}
            for add in missing:
                if add in s2:
                    continue
                ok = True
                for y in s2:
                    for z in pair_to_third.get((add, y), []):
                        if z in s2 and z != add and z != y:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    s2.add(add)
                    s = s2
                    improved = True
                    swapped = True
                    break
            if swapped:
                break
        if swapped:
            continue
        # Try removing an element that allows adding two others (net gain)
        lst = list(s)
        random.shuffle(lst)
        for rem in lst:
            s2 = s - {rem}
            # try to add as many as possible
            added_count = 0
            for add in missing:
                if add in s2:
                    continue
                ok = True
                for y in s2:
                    for z in pair_to_third.get((add, y), []):
                        if z in s2 and z != add and z != y:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    s2.add(add)
                    added_count += 1
            if added_count > 1:  # net gain
                s = s2
                improved = True
                break
    return s

def construct():
    global _START
    _START = time.time()
    random.seed(42)
    n = 180
    triples = _forbidden_triples(n)
    pair_to_third = _conflict_pairs(triples)

    def deadline_check():
        return _time_left() < 5.0

    best = set()

    # Multiple greedy orderings
    orderings = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        list(range(91, n + 1)) + list(range(1, 91)),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
    ]
    # Add some random orderings
    for _ in range(20):
        o = list(range(1, n + 1))
        random.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if deadline_check():
            break
        s = _greedy_construct(order, pair_to_third, n)
        if len(s) > len(best):
            best = set(s)

    # Local search from best and from several random starts
    candidates = [best]
    for _ in range(30):
        o = list(range(1, n + 1))
        random.shuffle(o)
        candidates.append(_greedy_construct(o, pair_to_third, n))

    for s0 in candidates:
        if deadline_check():
            break
        s = _local_search(s0, pair_to_third, n, max_iters=50000, deadline_check=deadline_check)
        if len(s) > len(best):
            best = set(s)

    # Final: try to add any remaining elements greedily
    for x in range(1, n + 1):
        if deadline_check():
            break
        if x not in best:
            ok = True
            for y in best:
                for z in pair_to_third.get((x, y), []):
                    if z in best and z != x and z != y:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best.add(x)

    # Ensure nonempty
    if not best:
        best = {180}

    return sorted(best)
# EVOLVE-BLOCK-END
