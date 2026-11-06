"""Structural rewrite: greedy set-packing over a rich candidate pool."""
import math
import random
import time

MAX_MOD = 128

def _gcd(a, b):
    a = abs(a); b = abs(b)
    while b:
        a, b = b, a % b
    return a

def _candidates_for(m):
    """Return a diverse set of candidate residues for modulus m."""
    cands = set()
    cands.add(0)
    cands.add(1)
    cands.add(m - 1)
    # random sample
    for _ in range(40):
        cands.add(random.randrange(m))
    return list(cands)

def construct():
    start = time.time()
    deadline = start + 70.0  # safety margin under 90s wall-clock

    best = [[7, 8]]  # seed fallback (score 1.0)

    # Moduli to consider: all integers 2..128
    moduli = list(range(2, MAX_MOD + 1))

    # Try multiple random seeds / orderings
    for trial in range(300):
        if time.time() > deadline:
            break

        # Random permutation of moduli
        order = moduli[:]
        random.shuffle(order)

        chosen = []  # list of (r, m)

        for m in order:
            if time.time() > deadline:
                break
            # Try candidates; pick first that is disjoint from all chosen
            cands = _candidates_for(m)
            # Try 0,1,m-1 first (deterministic), then random
            placed = False
            for r in cands:
                ok = True
                for (r2, m2) in chosen:
                    if _gcd(r - r2, m * m2) == 1:
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    placed = True
                    break
            # If no candidate worked, skip this modulus

        if len(chosen) > len(best):
            best = chosen[:]

    # Convert to required format
    result = [[r, m] for (r, m) in best]
    if not result:
        result = [[7, 8]]
    return result
