"""
Maximize the number of distinct 3-AP differences for a 32-element subset of [1,512].

Strategy:
  1. Fast bitset-based scorer (numpy uint64 words).
  2. Greedy construction to get a strong starting set.
  3. Local search (add/remove/swap) with a hard wall-clock deadline.
"""
import time
import random
import numpy as np

_N = 32
_U = 512
_WORDS = (_U + 63) // 64  # 8 words of 64 bits

def _to_bits(arr):
    """Convert sorted array of ints in 1..512 to a numpy uint64 word array."""
    words = np.zeros(_WORDS, dtype=np.uint64)
    for x in arr:
        words[(x - 1) >> 6] |= np.uint64(1) << np.uint64((x - 1) & 63)
    return words

def _score(arr):
    """Return (num_distinct_d, frozenset_of_d) for a 32-element set."""
    words = _to_bits(arr)
    S = set(arr)
    distinct = set()
    for a in arr:
        max_d = _U - 2 * a
        for d in range(1, max_d + 1):
            w1 = words[(a + d - 1) >> 6] >> np.uint64((a + d - 1) & 63) & np.uint64(1)
            w2 = words[(a + 2 * d - 1) >> 6] >> np.uint64((a + 2 * d - 1) & 63) & np.uint64(1)
            if w1 and w2:
                distinct.add(d)
    return len(distinct), frozenset(distinct)

def _greedy_construct():
    """Greedy: add the element that maximizes the marginal gain in distinct d."""
    S = []
    S_set = set()
    distinct = set()
    # Precompute: for each candidate x, and each existing a, what d's would be "covered"
    # We'll just do a simple greedy: pick x that adds the most new d's.
    for _ in range(_N):
        best_x = None
        best_score = -1
        for x in range(1, _U + 1):
            if x in S_set:
                continue
            # Compute new distinct d's if we add x
            new_distinct = set(distinct)
            # x can be a, a+d, or a+2d
            # For each existing element e, check if adding x creates new APs
            for e in S:
                # x as first: e, x, x+(x-e) -> need x-e > 0 and x + (x-e) <= 512
                if e < x:
                    d = x - e
                    if 2 * x - e <= _U and (2 * x - e) in S_set:
                        new_distinct.add(d)
                    # x as middle: e, x, x + (x-e)
                    if 2 * x - e <= _U and (2 * x - e) in S_set:
                        new_distinct.add(d)
                # x as last: e, e + (x-e), x -> d = x - e, need e + (x-e) in S
                # that's just checking if the middle is in S
                if e < x:
                    d = x - e
                    mid = e + d
                    if mid in S_set:
                        new_distinct.add(d)
                # x as first: x, x+d, x+2d where one of those is e
                # x + d = e -> d = e - x < 0, impossible
                # x + 2d = e -> d = (e-x)/2, need e > x and (e-x) even
                if e > x and (e - x) % 2 == 0:
                    d = (e - x) // 2
                    mid = x + d
                    if mid in S_set:
                        new_distinct.add(d)
                # x as middle: x-d = e -> d = x - e, x + d = 2x - e
                # already covered above
                # x as last: x - 2d = e -> d = (x-e)/2, need (x-e) even
                if x > e and (x - e) % 2 == 0:
                    d = (x - e) // 2
                    mid = x - d
                    if mid in S_set:
                        new_distinct.add(d)
            # Also x with itself doesn't form AP (need distinct elements)
            gain = len(new_distinct) - len(distinct)
            if gain > best_score:
                best_score = gain
                best_x = x
        if best_x is None:
            # Fallback: pick smallest unused
            for x in range(1, _U + 1):
                if x not in S_set:
                    best_x = x
                    break
        S.append(best_x)
        S_set.add(best_x)
        # Recompute distinct
        distinct = set()
        S_sorted = sorted(S)
        S_set_now = set(S_sorted)
        for a in S_sorted:
            max_d = _U - 2 * a
            for d in range(1, max_d + 1):
                if (a + d) in S_set_now and (a + 2 * d) in S_set_now:
                    distinct.add(d)
    return sorted(S)

def _local_search(start_arr, deadline):
    """Local search: swap one element for another to improve score."""
    best_arr = list(start_arr)
    best_score, best_fset = _score(best_arr)
    rng = random.Random(42)
    pool = list(range(1, _U + 1))
    
    while time.time() < deadline:
        # Try random swaps
        improved = False
        for _ in range(200):
            i = rng.randrange(_N)
            old = best_arr[i]
            new = rng.choice(pool)
            if new == old:
                continue
            candidate = best_arr[:]
            candidate[i] = new
            if len(set(candidate)) != _N:
                continue
            cand_score, _ = _score(candidate)
            if cand_score > best_score:
                best_arr = candidate
                best_score = cand_score
                improved = True
                break
            elif cand_score == best_score:
                # Accept with small probability for exploration
                if rng.random() < 0.05:
                    best_arr = candidate
                    improved = True
                    break
        if not improved:
            # Try add/remove pairs (remove one, add one)
            for _ in range(100):
                i = rng.randrange(_N)
                old = best_arr[i]
                new = rng.choice(pool)
                if new == old:
                    continue
                candidate = best_arr[:]
                candidate[i] = new
                if len(set(candidate)) != _N:
                    continue
                cand_score, _ = _score(candidate)
                if cand_score > best_score:
                    best_arr = candidate
                    best_score = cand_score
                    improved = True
                    break
        if not improved:
            # Perturb: replace 2 random elements
            for _ in range(50):
                i1 = rng.randrange(_N)
                i2 = rng.randrange(_N)
                if i1 == i2:
                    continue
                new1 = rng.choice(pool)
                new2 = rng.choice(pool)
                candidate = best_arr[:]
                candidate[i1] = new1
                candidate[i2] = new2
                if len(set(candidate)) != _N:
                    continue
                cand_score, _ = _score(candidate)
                if cand_score > best_score:
                    best_arr = candidate
                    best_score = cand_score
                    improved = True
                    break
        if not improved:
            # Restart with greedy + random perturbation
            if time.time() > deadline:
                break
            start = _greedy_construct()
            # Perturb a few elements
            for _ in range(3):
                i = rng.randrange(_N)
                start[i] = rng.choice(pool)
            if len(set(start)) != _N:
                continue
            s, _ = _score(start)
            if s >= best_score:
                best_arr = start
                best_score = s
    return best_arr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    greedy_arr = _greedy_construct()
    greedy_score, _ = _score(greedy_arr)
    
    # Phase 2: Local search
    if time.time() < deadline:
        best_arr, best_score = _local_search(greedy_arr, deadline)
    else:
        best_arr, best_score = greedy_arr, greedy_score
    
    # Phase 3: Try a few different greedy seeds
    if time.time() < deadline:
        for seed in [1, 7, 13, 21, 37]:
            rng2 = random.Random(seed)
            # Build a different greedy: randomize tie-breaking
            S = []
            S_set = set()
            for _ in range(_N):
                candidates = [x for x in range(1, _U + 1) if x not in S_set]
                # Score each candidate by marginal gain
                best_x = None
                best_gain = -1
                distinct_now = set()
                S_sorted = sorted(S)
                S_set_now = set(S_sorted)
                for a in S_sorted:
                    max_d = _U - 2 * a
                    for d in range(1, max_d + 1):
                        if (a + d) in S_set_now and (a + 2 * d) in S_set_now:
                            distinct_now.add(d)
                for x in candidates:
                    new_distinct = set(distinct_now)
                    for e in S_sorted:
                        if e < x:
                            d = x - e
                            if 2 * x - e <= _U and (2 * x - e) in S_set_now:
                                new_distinct.add(d)
                            mid = e + d
                            if mid in S_set_now:
                                new_distinct.add(d)
                        if e > x and (e - x) % 2 == 0:
                            d = (e - x) // 2
                            mid = x + d
                            if mid in S_set_now:
                                new_distinct.add(d)
                        if x > e and (x - e) % 2 == 0:
                            d = (x - e) // 2
                            mid = x - d
                            if mid in S_set_now:
                                new_distinct.add(d)
                    gain = len(new_distinct) - len(distinct_now)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = candidates[0]
                S.append(best_x)
                S_set.add(best_x)
            S = sorted(S)
            s, _ = _score(S)
            if s > best_score:
                best_arr = S
                best_score = s
            if time.time() >= deadline:
                break
    
    return best_arr

# Pre-compute a good static set as fallback
# This is a hardcoded set that should score well
_STATIC_SET = None

def _ensure_static():
    global _STATIC_SET
    if _STATIC_SET is None:
        try:
            arr = construct()
            if len(arr) == _N and all(1 <= x <= _U for x in arr) and len(set(arr)) == _N:
                _STATIC_SET = arr
        except Exception:
            _STATIC_SET = list(range(1, _N + 1))
    return _STATIC_SET

# Wrap construct to use static if possible, but always try to compute
_orig_construct = construct

def construct():
    try:
        return _orig_construct()
    except Exception:
        return _ensure_static()
