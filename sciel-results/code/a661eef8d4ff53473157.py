import math
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = 130.0  # safety margin under the 150s wall-clock kill

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _pair_ok(m1, r1, m2, r2):
    g = gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            if not _pair_ok(m1, r1, m2, r2):
                return False
    return True

def _score(classes):
    return sum(1.0 / m for m, _ in classes)

def _find_residue(classes, m):
    """Return a residue r for modulus m that is disjoint from all existing, else None."""
    for r in range(m):
        ok = True
        for (mi, ri) in classes:
            if (r % gcd(m, mi)) == (ri % gcd(m, mi)):
                ok = False
                break
        if ok:
            return r
    return None

def _common_factor(K):
    multiples = [m for m in range(20, 201) if m % K == 0]
    if not multiples:
        return []
    k = min(K, len(multiples))
    sel = sorted(multiples)[:k]
    return [[m, idx] for idx, m in enumerate(sel)]

def _best_common_factor():
    best, best_s = [], -1.0
    for K in range(1, 201):
        c = _common_factor(K)
        if not c:
            continue
        if not _is_valid(c):
            continue
        s = _score(c)
        if s > best_s:
            best_s, best = s, c
    return best

def _add_pass(classes):
    """Add every missing modulus that has a valid residue. Returns True if anything added."""
    added = False
    used = set(m for m, _ in classes)
    for m in range(20, 201):
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
            added = True
    return added

def _reassign_pass(classes, rng, max_attempts=4000):
    """Try reassigning one existing residue to enable adding a missing modulus."""
    n = len(classes)
    if n == 0:
        return False
    used = set(m for m, _ in classes)
    missing = [m for m in range(20, 201) if m not in used]
    if not missing:
        return False
    attempts = 0
    for m in missing:
        for i in range(n):
            if attempts >= max_attempts or _time_left() < 8:
                return False
            mi = classes[i][0]
            old = classes[i][1]
            for _ in range(6):  # a few random candidate residues
                attempts += 1
                if attempts >= max_attempts or _time_left() < 8:
                    break
                rnew = rng.randrange(mi)
                if rnew == old:
                    continue
                classes[i][1] = rnew
                # only re-check pairs involving i
                ok = True
                for j in range(n):
                    if j == i:
                        continue
                    mj, rj = classes[j]
                    if not _pair_ok(mi, rnew, mj, rj):
                        ok = False
                        break
                if not ok:
                    classes[i][1] = old
                    continue
                r = _find_residue(classes, m)
                if r is not None:
                    classes.append([m, r])
                    return True
                classes[i][1] = old
    return False

def _local_search(classes, budget, rng):
    t0 = time.time()
    while (time.time() - t0) < budget and _time_left() > 8:
        progress = _add_pass(classes)
        if progress:
            continue
        if _reassign_pass(classes, rng):
            continue
        break  # local optimum
    return classes

def construct():
    rng = random.Random(20240517)
    best, best_score = None, -1.0

    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]

    # 1) Best single common-factor seed, then hill-climb.
    seed = _best_common_factor()
    if seed:
        c = [list(x) for x in seed]
        c = _local_search(c, min(40.0, max(5.0, _time_left() - 20)), rng)
        consider(c)

    # 2) Random-restart hill-climbs from random-order greedy starts.
    while _time_left() > 15:
        order = list(range(20, 201))
        rng.shuffle(order)
        c = []
        used = set()
        for m in order:
            if m in used:
                continue
            r = _find_residue(c, m)
            if r is not None:
                c.append([m, r])
                used.add(m)
            if _time_left() < 15:
                break
        c = _local_search(c, min(6.0, max(1.0, _time_left() - 15)), rng)
        consider(c)

    # Fallback (should not be reached).
    if best is None:
        best = [[20 * j, j - 1] for j in range(1, 11)]
    return best
