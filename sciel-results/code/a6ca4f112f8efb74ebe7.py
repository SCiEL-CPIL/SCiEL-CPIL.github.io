"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

_N = 128
_DEADLINE = 80.0  # safety margin under 90s kill

def _lcm(a, b):
    return a // math.gcd(a, b) * b

def _bad_triple_exists(selected, x):
    """Check if adding x creates a bad triple with two existing elements."""
    s = selected
    n = len(s)
    for i in range(n):
        a = s[i]
        la = _lcm(a, x)
        for j in range(i + 1, n):
            b = s[j]
            if la == _lcm(a, b) == _lcm(b, x):
                return True
    return False

def _validate(selected):
    s = list(selected)
    for a, b, c in combinations(s, 3):
        if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
            return False
    return True

def _greedy(order):
    selected = []
    for x in order:
        if not _bad_triple_exists(selected, x):
            selected.append(x)
    return selected

def _local_search(selected):
    """Try to add any missing element, or swap to grow size."""
    selected = list(selected)
    selset = set(selected)
    improved = True
    while improved:
        improved = False
        # try adding
        for x in range(1, _N + 1):
            if x not in selset and not _bad_triple_exists(selected, x):
                selected.append(x)
                selset.add(x)
                improved = True
                break
        # try swap: remove one, add one to net +? can't net gain by swap; skip
    return selected

def construct():
    t0 = time.time()
    base = list(range(1, _N + 1))
    orders = [
        base,
        list(reversed(base)),
        sorted(base, key=lambda v: -v.bit_count()),
        sorted(base, key=lambda v: v.bit_count()),
        sorted(base, key=lambda v: -math.gcd(v, 128)),
    ]
    # a few random orders
    rng = random.Random(12345)
    for _ in range(4):
        o = base[:]
        rng.shuffle(o)
        orders.append(o)

    best = []
    for o in orders:
        if time.time() - t0 > _DEADLINE - 5:
            break
        cand = _greedy(o)
        cand = _local_search(cand)
        if len(cand) > len(best):
            best = cand

    # time-bounded local search rounds on best
    best_set = set(best)
    while time.time() - t0 < _DEADLINE - 5:
        # try adding any
        added = False
        for x in range(1, _N + 1):
            if x not in best_set and not _bad_triple_exists(best, x):
                best.append(x)
                best_set.add(x)
                added = True
                break
        # try swap to enable more: remove r, see if we can add two
        for r in list(best):
            if time.time() - t0 > _DEADLINE - 5:
                break
            trial = [v for v in best if v != r]
            trial_set = set(trial)
            gained = 0
            for x in range(1, _N + 1):
                if x not in trial_set and not _bad_triple_exists(trial, x):
                    trial.append(x)
                    trial_set.add(x)
                    gained += 1
                    break
            if gained + 0 > 0 and len(trial) > len(best):
                best = trial
                best_set = set(best)
        if not added and time.time() - t0 > _DEADLINE - 10:
            break

    if not best:
        best = [1]
    return best
# EVOLVE-BLOCK-END
