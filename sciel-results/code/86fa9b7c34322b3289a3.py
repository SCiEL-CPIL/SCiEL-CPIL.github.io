"""Rewritten program for the pair-sum-divisibility task."""
import time
from math import gcd

# EVOLVE-BLOCK-START

def _conflicts(a, b):
    g = gcd(a, b)
    m = a // g
    n = b // g
    # conflict iff (m+n) divides g
    return (g % (m + n)) == 0

def _is_compatible(x, chosen):
    for y in chosen:
        if _conflicts(x, y):
            return False
    return True

def _greedy_build(ordered):
    chosen = []
    chosen_set = set()
    for x in ordered:
        if x in chosen_set:
            continue
        ok = True
        for y in chosen:
            if _conflicts(x, y):
                ok = False
                break
        if ok:
            chosen.append(x)
            chosen_set.add(x)
    return chosen

def construct():
    deadline = time.time() + 50.0  # safety margin under 90s kill

    odds = list(range(1, 257, 2))          # 128 odds, provably conflict-free
    evens = list(range(2, 257, 2))

    # Build a conflict map: for each even, which odds does it conflict with?
    # (odds among themselves never conflict)
    even_conflicts_odd = {}
    for e in evens:
        bad = set()
        for o in odds:
            if _conflicts(e, o):
                bad.add(o)
        even_conflicts_odd[e] = bad

    # Greedy: add evens that conflict with the fewest existing members.
    # Start from all odds, then try to add evens sorted by (conflicts with odds, value).
    def greedy():
        chosen = list(odds)
        chosen_set = set(odds)
        remaining = list(evens)
        # sort by number of conflicts with current chosen (odds)
        remaining.sort(key=lambda e: (len(even_conflicts_odd[e]), e))
        for e in remaining:
            if e in chosen_set:
                continue
            # check conflicts against all currently chosen
            ok = True
            for y in chosen:
                if _conflicts(e, y):
                    ok = False
                    break
            if ok:
                chosen.append(e)
                chosen_set.add(e)
        return chosen

    best = greedy()
    best_size = len(best)

    # Try several different orderings to improve greedy quality
    import random
    rng = random.Random(12345)
    for trial in range(2000):
        if time.time() > deadline - 5.0:
            break
        # perturb ordering: shuffle evens, keep odds first
        order = list(odds)
        ev = list(evens)
        rng.shuffle(ev)
        order.extend(ev)
        cand = _greedy_build(order)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)

    # Local search: try to swap a chosen element with a non-chosen one to gain size
    chosen_set = set(best)
    nonchosen = [x for x in range(1, 257) if x not in chosen_set]
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for out in list(best):
            # try removing 'out' and adding some nonchosen 'in'
            without = [x for x in best if x != out]
            without_set = set(without)
            for inp in nonchosen:
                if inp in without_set:
                    continue
                if not _is_compatible(inp, without):
                    continue
                # size unchanged; only useful if it enables more later. Skip for now.
                pass
            # try just removing out to see if we can add two
            pass
        # Simpler hill climb: try adding a nonchosen element after removing one chosen
        for out in list(best):
            if time.time() > deadline - 2.0:
                break
            without = [x for x in best if x != out]
            without_set = set(without)
            # find a nonchosen that fits
            for inp in nonchosen:
                if inp in without_set:
                    continue
                if not _is_compatible(inp, without):
                    continue
                newbest = without + [inp]
                # try to add any other nonchosen
                extra = 0
                for z in nonchosen:
                    if z in set(newbest):
                        continue
                    if _is_compatible(z, newbest):
                        newbest.append(z)
                        extra += 1
                if len(newbest) > len(best):
                    best = newbest
                    improved = True
                    break
            if improved:
                break

    return best
# EVOLVE-BLOCK-END
