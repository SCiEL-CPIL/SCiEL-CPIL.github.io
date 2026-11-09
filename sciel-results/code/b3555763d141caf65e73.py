"""Rewritten: fast bitmask subset-sum validity + local-search optimizer."""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 82.0  # safety margin under 90s wall-clock

def _subset_sum_reach(bits, target):
    """Return True iff some non-empty subset of bits sums exactly to target."""
    if target <= 0:
        return False
    seen = 1  # bit 0 set: sum 0 reachable
    for b in bits:
        seen |= (seen << b)
        # early exit: once we can reach target, stop
        if (seen >> target) & 1:
            return True
    return (seen >> target) & 1

def _is_valid(sel):
    """Check no element is the mean of any subset (>=2) of the OTHER elements."""
    n = len(sel)
    for i in range(n):
        a = sel[i]
        bits = [sel[j] for j in range(n) if j != i]
        for k in range(2, n):
            if _subset_sum_reach(bits, k * a):
                return False
    return True

def _valid_add(sel, x):
    """Check whether adding x keeps the set valid (re-check all elements)."""
    n = len(sel)
    bits = list(sel)
    # check existing elements against new subset (which now includes x)
    for i in range(n):
        a = sel[i]
        oth = [y for y in sel if y != a] + [x]
        for k in range(2, n + 1):
            if _subset_sum_reach(oth, k * a):
                return False
    # check x against existing elements
    for k in range(2, n + 1):
        if _subset_sum_reach(sel, k * x):
            return False
    return True

def construct():
    start = time.time()
    rng = random.Random(12345)

    # Phase 1: greedy seed (ascending)
    sel = []
    for x in range(1, 513):
        if _valid_add(sel, x):
            sel.append(x)
            if len(sel) >= 48:
                break
    best = list(sel)

    # Phase 2: local search with add/remove/swap moves
    def time_left():
        return DEADLINE - (time.time() - start)

    # Build candidate pool: values not currently in sel
    present = set(sel)
    pool = [v for v in range(1, 513) if v not in present]
    rng.shuffle(pool)

    improved = True
    while improved and time_left() > 0.5:
        improved = False
        # Try add moves
        for x in pool:
            if time_left() < 0.5:
                break
            if _valid_add(sel, x):
                new_sel = sel + [x]
                if _is_valid(new_sel):
                    sel = new_sel
                    present.add(x)
                    pool.remove(x)
                    improved = True
                    if len(sel) >= 48:
                        best = list(sel)
                        return best
        # Try swap moves: replace one element with a pool element to enable growth
        if len(sel) < 48 and time_left() > 1.0:
            n = len(sel)
            # sample some swap attempts
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time_left() > 1.0:
                attempts += 1
                idx = rng.randrange(n)
                old = sel[idx]
                cand = rng.choice(pool) if pool else None
                if cand is None or cand == old:
                    continue
                new_sel = sel[:idx] + [cand] + sel[idx+1:]
                if _is_valid(new_sel):
                    sel = new_sel
                    present.discard(old)
                    present.add(cand)
                    if old in pool:
                        pool.remove(old)
                    if cand not in pool:
                        pool.append(cand)
                    improved = True
        # If no growth, try a fresh randomized restart occasionally
        if not improved and len(sel) < 48 and time_left() > 2.0:
            # Randomized greedy restart
            vals = list(range(1, 513))
            rng.shuffle(vals)
            new_sel = []
            for x in vals:
                if _valid_add(new_sel, x):
                    new_sel.append(x)
                if len(new_sel) > len(sel):
                    pass
            if len(new_sel) > len(sel):
                sel = new_sel
                present = set(sel)
                pool = [v for v in range(1, 513) if v not in present]
                rng.shuffle(pool)
                improved = True

    if len(sel) > len(best):
        best = sel
    return best
# EVOLVE-BLOCK-END
