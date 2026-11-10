"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

def _factorize(num):
    """Return tuple of prime->exponent for num (indexed by _PRIMES)."""
    factors = [0] * _NUM_PRIMES
    x = num
    for i, p in enumerate(_PRIMES):
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[i] = cnt
    return tuple(factors)

# Precompute factorizations as tuples
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _vec_add(a, b):
    return tuple(a[i] + b[i] for i in range(_NUM_PRIMES))

def _can_add_to_sums(sums_set, v):
    """Check if vector v can be added to sums_set without creating duplicates."""
    for s in sums_set:
        ns = _vec_add(s, v)
        if ns in sums_set:
            return False
    return True

def _add_to_sums(sums_set, v):
    """Add all sums with v to sums_set in-place. Assumes _can_add_to_sums passed."""
    new_sums = set()
    for s in sums_set:
        new_sums.add(_vec_add(s, v))
    sums_set.update(new_sums)

def _build_sums(nums):
    """Build the full subset-sum set for a list of numbers."""
    sums_set = {(0,) * _NUM_PRIMES}
    for n in nums:
        _add_to_sums(sums_set, _FACTS[n])
    return sums_set

def _validate_by_products(nums):
    """Explicit check: all 2^k subset products distinct (ground truth)."""
    k = len(nums)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _greedy_build(ordering):
    """Greedy: start with 12 primes, add candidates in given ordering."""
    primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    current = list(primes_in_range)
    used = set(current)
    sums_set = {(0,) * _NUM_PRIMES}
    for p in primes_in_range:
        _add_to_sums(sums_set, _FACTS[p])

    for cand in ordering:
        if len(current) >= 18:
            break
        if cand in used:
            continue
        v = _FACTS[cand]
        if _can_add_to_sums(sums_set, v):
            current.append(cand)
            used.add(cand)
            _add_to_sums(sums_set, v)
    return current

def _local_search(current, deadline):
    """Try to improve current set via swaps: replace a member with a candidate."""
    current = list(current)
    used = set(current)
    candidates = [n for n in range(2, 41) if n not in used]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            for cand in candidates:
                trial = current[:i] + current[i+1:] + [cand]
                # quick incremental check
                trial_set = set(trial)
                if len(trial_set) != len(trial):
                    continue
                # rebuild sums and check
                sums_set = {(0,) * _NUM_PRIMES}
                ok = True
                for n in trial:
                    v = _FACTS[n]
                    if not _can_add_to_sums(sums_set, v):
                        ok = False
                        break
                    _add_to_sums(sums_set, v)
                if ok:
                    # accept if it lets us later add more (heuristic: accept any valid swap
                    # that changes the set, to explore)
                    current = trial
                    used = set(current)
                    candidates = [n for n in range(2, 41) if n not in used]
                    improved = True
                    break
            if improved:
                break
    return current

def _randomized_search(deadline, rng):
    """Randomized greedy with local search polish."""
    primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    candidates = [n for n in range(4, 41) if n not in set(primes_in_range)]
    best = None
    best_len = 0
    while time.time() < deadline:
        order = list(candidates)
        rng.shuffle(order)
        res = _greedy_build(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
        if best_len >= 18:
            break
        # local search polish
        res2 = _local_search(res, deadline)
        if len(res2) > best_len:
            best = res2
            best_len = len(res2)
        if best_len >= 18:
            break
    return best if best is not None else []

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    rng = random.Random(12345)

    # 1) Deterministic greedy with several heuristics (fast, near-instant)
    candidates = [n for n in range(4, 41) if n not in {2,3,5,7,11,13,17,19,23,29,31,37}]

    heuristics = [
        lambda n: (sum(1 for x in _FACTS[n] if x > 0), sum(_FACTS[n]), n),
        lambda n: (sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0), n),
        lambda n: (n, sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0)),
        lambda n: (-n, sum(_FACTS[n]), sum(1 for x in _FACTS[n] if x > 0)),
        lambda n: (sum(x*x for x in _FACTS[n]), sum(_FACTS[n]), n),
    ]

    best = None
    best_len = 0
    for h in heuristics:
        if time.time() >= deadline:
            break
        order = sorted(candidates, key=h)
        res = _greedy_build(order)
        if len(res) > best_len:
            best = res
            best_len = len(res)
        if best_len >= 18:
            break

    # 2) Randomized search + local search to close the gap
    if best_len < 18 and time.time() < deadline:
        res = _randomized_search(deadline, rng)
        if len(res) > best_len:
            best = res
            best_len = len(res)

    if best is None or len(best) == 0:
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # 3) Final ground-truth validation; if it fails, fall back to a verified subset
    if len(best) >= 2:
        if not _validate_by_products(best):
            # shrink until valid (should not happen, but be safe)
            for k in range(len(best), 11, -1):
                sub = best[:k]
                if _validate_by_products(sub):
                    best = sub
                    break

    # Trim to at most 18
    if len(best) > 18:
        best = best[:18]
        if not _validate_by_products(best):
            for k in range(18, 11, -1):
                sub = best[:k]
                if _validate_by_products(sub):
                    best = sub
                    break

    return [int(x) for x in best]

# EVOLVE-BLOCK-END
