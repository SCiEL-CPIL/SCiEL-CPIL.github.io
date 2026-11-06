import time
import random
import numpy as np

BASE = 51975
DEADLINE = time.time() + 145.0

def _divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    ds.sort()
    return ds

def construct():
    global DEADLINE
    start = time.time()
    divs = _divisors(BASE)
    moduli = [d for d in divs if d > 1]
    
    masks = {}
    for m in moduli:
        masks[m] = np.arange(BASE) % m

    def greedy(order):
        covered = np.zeros(BASE, dtype=np.uint8)
        res = {}
        for m in order:
            if time.time() > DEADLINE:
                break
            mask = masks[m]
            unc = 1 - covered
            best_r = 0
            best_c = -1
            for r in range(m):
                c = int(unc[mask == r].sum())
                if c > best_c:
                    best_c = c
                    best_r = r
            res[m] = best_r
            covered[mask == best_r] = 1
        return res, int((1 - covered).sum())

    best_res = None
    best_score = float('inf')
    
    # Large to small
    res, sc = greedy(sorted(moduli, reverse=True))
    if sc < best_score:
        best_score = sc
        best_res = res

    # Small to large
    if time.time() < DEADLINE:
        res, sc = greedy(sorted(moduli))
        if sc < best_score:
            best_score = sc
            best_res = res

    # Random orderings
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > DEADLINE:
            break
        order = moduli[:]
        rng.shuffle(order)
        res, sc = greedy(order)
        if sc < best_score:
            best_score = sc
            best_res = res

    out = [[m, int(best_res[m])] for m in moduli if m in best_res]
    return out
