import time
import random
import itertools

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    n = len(coefs)
    cnt = 0
    for i in range(n):
        a = coefs[i]
        if a == 0:
            continue
        for j in range(i, n):
            v = a * a if i == j else 2 * a * coefs[j]
            if v != 0:
                cnt += 1
                break
    return cnt

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _try_place(best, support, rng):
    # assign coefficients to the given support, optimize to reduce square nz
    n = len(support)
    best_local = None
    best_score = 10**9
    for _ in range(400):
        if time.time() > _DEADLINE:
            break
        coefs = [0] * 41
        for k, d in enumerate(support):
            c = rng.choice([-2, -1, 1, 2])
            coefs[d] = c
        # ensure final (highest degree) nonzero
        hi = max(support)
        if coefs[hi] == 0:
            coefs[hi] = 1
        sc = _sq_nonzero_count(coefs)
        if sc < best_score:
            best_score = sc
            best_local = coefs[:]
        # local perturbation
        if best_local is not None:
            cand = best_local[:]
            d = support[rng.randrange(n)]
            old = cand[d]
            new = old + rng.choice([-2, -1, 1, 2])
            if -64 <= new <= 64 and new != 0:
                cand[d] = new
            # occasionally zero out a non-top term and add a new small term elsewhere
            if rng.random() < 0.2:
                d2 = support[rng.randrange(n)]
                if d2 != hi and cand[d2] != 0:
                    cand[d2] = 0
                    # re-add somewhere in support to keep 12 terms
                    for dd in support:
                        if cand[dd] == 0 and dd != hi:
                            cand[dd] = rng.choice([-1, 1])
                            break
            if _valid(cand):
                sc2 = _sq_nonzero_count(cand)
                if sc2 < best_score:
                    best_score = sc2
                    best_local = cand[:]
    if best_local is not None and _valid(best_local):
        sc = _sq_nonzero_count(best_local)
        if sc < best[0]:
            best[0] = sc
            best[1] = best_local[:]
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 80.0
    rng = random.Random(12345)
    best = [10**9, None]

    # Structural supports: choose 12 distinct degrees in [0,40], force degree 40 present
    def make_support():
        pool = list(range(0, 40))
        rng.shuffle(pool)
        chosen = sorted(pool[:11]) + [40]
        return chosen

    # Try several randomized supports
    for t in range(60):
        if time.time() > _DEADLINE:
            break
        sup = make_support()
        _try_place(best, sup, rng)

    # Also try arithmetic-structured supports (large gaps -> natural zero slots)
    for start in range(0, 8):
        for gap in range(3, 12):
            sup = []
            d = start
            while d <= 40 and len(sup) < 12:
                sup.append(d)
                d += gap
            if len(sup) < 12:
                continue
            if 40 not in sup:
                sup = sup[:-1] + [40]
                sup = sorted(set(sup))
                if len(sup) != 12:
                    continue
            _try_place(best, sup, rng)
            if time.time() > _DEADLINE:
                break

    if best[1] is None:
        # fallback: a valid 12-term polynomial
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs

    return best[1]
