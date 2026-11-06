import time
import numpy as np
from fractions import Fraction
from math import isqrt

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_PRIME_SET = set(_PRIMES)
_MAX_L = 100000

def _build_mask(classes, cap):
    """Build coverage mask for 1..cap using NumPy."""
    n = cap + 1
    covered = np.zeros(n, dtype=bool)
    for p, r in classes:
        start = r if r >= 1 else p
        if start <= cap:
            covered[start::p] = True
    return covered

def _coverage_length(covered_mask, cap):
    """Find largest L such that 1..L all covered."""
    if cap < 1:
        return 0
    c = covered_mask[1:cap+1]
    if c.all():
        return cap
    idx = np.argmax(~c)
    return int(idx)

def _exact_sum(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _valid(classes):
    seen = set()
    s = Fraction(0, 1)
    for p, r in classes:
        if p in seen:
            return False
        if p < 2 or p > 1000 or p not in _PRIME_SET:
            return False
        if r < 0 or r >= p:
            return False
        seen.add(p)
        s += Fraction(1, p)
        if s > 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_L = 0
    best_classes = []

    def _time_left():
        return deadline - time.time()

    # --- Phase 1: Greedy set-cover to get initial solution ---
    def greedy_initial():
        nonlocal best_L, best_classes
        classes = []
        used = set()
        budget = Fraction(1)
        # Work on a moderate cap first for speed
        cap = 5000
        covered = np.zeros(cap + 1, dtype=bool)

        while _time_left() > 30:
            # Find first gap
            if covered[1:].all():
                break
            first_gap = int(np.argmax(~covered[1:])) + 1

            best_p = None
            best_r = None
            best_gain = 0

            for p in _PRIMES:
                if p in used:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                r = first_gap % p
                start = r if r >= 1 else p
                if start > cap:
                    continue
                # Count new coverage in window [first_gap, first_gap+2000]
                end = min(first_gap + 2000, cap)
                if end <= start:
                    continue
                mask = np.arange(start, end + 1, p)
                mask = mask[mask <= end]
                if len(mask) == 0:
                    continue
                new_count = int(np.sum(~covered[mask]))
                # Prefer smaller primes (cheaper) with good coverage
                score = new_count * 1000 - p
                if score > best_gain:
                    best_gain = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            classes.append([best_p, best_r])
            used.add(best_p)
            budget -= Fraction(1, best_p)
            start = best_r if best_r >= 1 else best_p
            if start <= cap:
                covered[start::best_p] = True

        # Evaluate on full cap
        if classes:
            full_mask = _build_mask(classes, _MAX_L)
            L = _coverage_length(full_mask, _MAX_L)
            if L > best_L:
                best_L = L
                best_classes = [list(c) for c in classes]
        return classes

    initial_classes = greedy_initial()
    current_classes = [list(c) for c in initial_classes] if initial_classes else []
    current_L = best_L

    # --- Phase 2: Local search with 1-opt (add) and 2-opt (swap) ---
    def local_search(classes, max_iters=5000):
        nonlocal best_L, best_classes
        if not classes:
            return
        classes = [list(c) for c in classes]
        no_improve = 0
        cap = _MAX_L

        for iteration in range(max_iters):
            if _time_left() < 5:
                break

            # Recompute mask
            covered = _build_mask(classes, cap)
            L = _coverage_length(covered, cap)
            if L > best_L:
                best_L = L
                best_classes = [list(c) for c in classes]

            if L >= cap:
                break

            budget = 1 - _exact_sum(classes)
            if L == 0:
                break

            first_gap = L + 1
            if first_gap > cap:
                break

            improved = False

            # Try 1-opt: add a new class
            if budget > 0:
                for p in _PRIMES:
                    if any(c[0] == p for c in classes):
                        continue
                    cost = Fraction(1, p)
                    if budget < cost:
                        continue
                    # Try residue that covers first_gap
                    r = first_gap % p
                    start = r if r >= 1 else p
                    if start > cap:
                        continue
                    # Quick check: does this help?
                    new_mask = covered.copy()
                    new_mask[start::p] = True
                    new_L = _coverage_length(new_mask, cap)
                    if new_L > L:
                        classes.append([p, r])
                        L = new_L
                        improved = True
                        break

            # Try 2-opt: swap one class for another
            if not improved:
                for i in range(len(classes)):
                    if _time_left() < 5:
                        break
                    p_old, r_old = classes[i]
                    # Remove class i
                    remaining = classes[:i] + classes[i+1:]
                    new_budget = 1 - _exact_sum(remaining)
                    if new_budget <= 0:
                        continue
                    rem_mask = _build_mask(remaining, cap)
                    rem_L = _coverage_length(rem_mask, cap)
                    if rem_L < L - 100:  # Don't remove if it hurts too much
                        continue

                    # Try adding a different prime
                    for p_new in _PRIMES:
                        if p_new == p_old:
                            continue
                        if any(c[0] == p_new for c in remaining):
                            continue
                        cost = Fraction(1, p_new)
                        if new_budget < cost:
                            continue
                        # Try residue covering the first gap of remaining
                        if rem_L < cap:
                            gap = rem_L + 1
                        else:
                            gap = L + 1
                        r_new = gap % p_new
                        start = r_new if r_new >= 1 else p_new
                        if start > cap:
                            continue
                        new_mask = rem_mask.copy()
                        new_mask[start::p_new] = True
                        new_L = _coverage_length(new_mask, cap)
                        if new_L > L:
                            classes[i] = [p_new, r_new]
                            L = new_L
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                no_improve += 1
                if no_improve > 50:
                    # Perturbation: randomly change a residue
                    if len(classes) > 0:
                        idx = np.random.randint(len(classes))
                        p, r = classes[idx]
                        classes[idx] = [p, np.random.randint(p)]
                        no_improve = 0
            else:
                no_improve = 0

        # Final evaluation
        if classes:
            final_mask = _build_mask(classes, cap)
            final_L = _coverage_length(final_mask, cap)
            if final_L > best_L:
                best_L = final_L
                best_classes = [list(c) for c in classes]

    # Run local search
    if current_classes and _time_left() > 10:
        local_search(current_classes)

    # --- Phase 3: Try alternative initial solutions and local search ---
    if _time_left() > 30:
        # Try with different prime orderings
        for seed_order in [
            sorted(_PRIMES, key=lambda p: p),
            sorted(_PRIMES, key=lambda p: -p),
            sorted(_PRIMES, key=lambda p: abs(p - 500)),
        ]:
            if _time_left() < 15:
                break
            # Quick greedy with this order
            classes = []
            used = set()
            budget = Fraction(1)
            cap = 3000
            covered = np.zeros(cap + 1, dtype=bool)

            while _time_left() > 10:
                if covered[1:].all():
                    break
                first_gap = int(np.argmax(~covered[1:])) + 1
                best_p = None
                best_r = None
                best_score = -1
                for p in seed_order:
                    if p in used:
                        continue
                    cost = Fraction(1, p)
                    if budget < cost:
                        continue
                    r = first_gap % p
                    start = r if r >= 1 else p
                    if start > cap:
                        continue
                    end = min(first_gap + 1000, cap)
                    if end <= start:
                        continue
                    mask = np.arange(start, end + 1, p)
                    mask = mask[mask <= end]
                    if len(mask) == 0:
                        continue
                    new_count = int(np.sum(~covered[mask]))
                    score = new_count * 1000 - p
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_r = r
                if best_p is None:
                    break
                classes.append([best_p, best_r])
                used.add(best_p)
                budget -= Fraction(1, best_p)
                start = best_r if best_r >= 1 else best_p
                if start <= cap:
                    covered[start::best_p] = True

            if classes:
                full_mask = _build_mask(classes, _MAX_L)
                L = _coverage_length(full_mask, _MAX_L)
                if L > best_L:
                    best_L = L
                    best_classes = [list(c) for c in classes]
                # Local search on this
                if L >= best_L - 50 and _time_left() > 10:
                    local_search(classes, max_iters=2000)

    # Ensure valid fallback
    if not best_classes or not _valid(best_classes):
        primes_13_plus = [p for p in _PRIMES if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
        # Validate
        if not _valid(best_classes):
            best_classes = [[13, 1]]

    return best_classes
