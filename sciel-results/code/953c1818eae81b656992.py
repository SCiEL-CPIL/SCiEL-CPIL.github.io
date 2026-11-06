from math import gcd
import time
import random

_START = time.time()
_DEADLINE = 140.0  # safety margin under 150s


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


def _greedy_build(order, classes=None):
    """Build a solution by processing moduli in the given order."""
    if classes is None:
        classes = []
    used = set()
    for m in order:
        if m in used:
            continue
        r = _find_residue(classes, m)
        if r is not None:
            classes.append([m, r])
            used.add(m)
    return classes


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


def _reassign_pass(classes, rng, max_attempts=2000):
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
            if attempts >= max_attempts or _time_left() < 10:
                return False
            mi = classes[i][0]
            old = classes[i][1]
            for _ in range(4):
                attempts += 1
                if attempts >= max_attempts or _time_left() < 10:
                    break
                rnew = rng.randrange(mi)
                if rnew == old:
                    continue
                classes[i][1] = rnew
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
    """Iteratively add missing moduli and reassign residues until no progress."""
    t0 = time.time()
    while (time.time() - t0) < budget and _time_left() > 10:
        progress = _add_pass(classes)
        if progress:
            continue
        if _reassign_pass(classes, rng):
            continue
        break
    return classes


def _branch_and_bound(classes, budget, rng):
    """
    Branch-and-bound MWIS-style optimization.
    Try to improve by swapping out low-value moduli for combinations of higher-value ones.
    """
    t0 = time.time()
    used = set(m for m, _ in classes)
    missing = [m for m in range(20, 201) if m not in used]
    
    # Sort missing by 1/m descending (most valuable first)
    missing.sort(key=lambda m: -1.0 / m)
    
    best_score = _score(classes)
    best_classes = [list(c) for c in classes]
    
    # Try removing one modulus and adding two or more missing ones
    for i in range(len(classes)):
        if _time_left() < 15 or (time.time() - t0) > budget:
            break
        
        mi, ri = classes[i]
        # Temporarily remove this modulus
        saved = classes.pop(i)
        
        # Try to add missing moduli in order of value
        added = []
        for m in missing:
            if m == mi:
                continue
            r = _find_residue(classes, m)
            if r is not None:
                classes.append([m, r])
                added.append([m, r])
        
        # Check if this improvement is better
        if len(added) >= 2:
            s = _score(classes)
            if s > best_score:
                best_score = s
                best_classes = [list(c) for c in classes]
        
        # Restore
        for a in reversed(added):
            classes.pop()
        classes.insert(i, saved)
    
    return best_classes


def construct():
    rng = random.Random(20240517)
    best = None
    best_score = -1.0
    
    def consider(classes):
        nonlocal best, best_score
        if classes and _is_valid(classes):
            s = _score(classes)
            if s > best_score:
                best_score = s
                best = [list(c) for c in classes]
    
    # Strategy 1: Even moduli (current program's approach)
    if _time_left() > 20:
        even_moduli = [m for m in range(20, 201) if m % 2 == 0]
        c1 = _greedy_build(even_moduli)
        c1 = _local_search(c1, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c1)
    
    # Strategy 2: Multiples of 3
    if _time_left() > 20:
        mult3 = [m for m in range(20, 201) if m % 3 == 0]
        c2 = _greedy_build(mult3)
        c2 = _local_search(c2, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c2)
    
    # Strategy 3: All moduli in ascending order
    if _time_left() > 20:
        all_moduli = list(range(20, 201))
        c3 = _greedy_build(all_moduli)
        c3 = _local_search(c3, min(10.0, max(2.0, _time_left() - 15)), rng)
        consider(c3)
    
    # Strategy 4: Random restarts
    restarts = 0
    while _time_left() > 25 and restarts < 50:
        order = list(range(20, 201))
        rng.shuffle(order)
        c = _greedy_build(order)
        c = _local_search(c, min(5.0, max(1.0, _time_left() - 20)), rng)
        consider(c)
        restarts += 1
    
    # Strategy 5: Branch-and-bound improvement on the best solution
    if best is not None and _time_left() > 30:
        c5 = [list(c) for c in best]
        c5 = _branch_and_bound(c5, min(40.0, max(5.0, _time_left() - 20)), rng)
        consider(c5)
    
    # Fallback
    if best is None:
        best = [[20 * j, j - 1] for j in range(1, 11)]
    
    return best
