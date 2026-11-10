"""Structural change: incremental scoring + diverse seeds + wider neighborhood.
We precompute distinct-prime-factor bitmasks (via a small prime table) for every
possible pairwise sum (2..20000). Then a full score of 12 numbers is just the
popcount of the OR of 66 precomputed masks. Local search can then evaluate
hundreds of thousands of moves per second, and we can afford many diverse
starting points (multiples of 6, 12, powers of 2, etc.) plus a generic
hill-climb to shave the raw objective."""
import time
import math
import random

_START_TIME = time.time()
_DEADLINE_SECONDS = 82.0
MAX_VAL = 10000
N = 12


def _elapsed():
    return time.time() - _START_TIME


def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS


# ---------- prime factorization via small-prime table ----------
# We only need to factor numbers up to 2*MAX_VAL = 20000.
# Precompute the distinct-prime-factor set for each n in [2, 20000].
_MAX_SUM = 2 * MAX_VAL
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
# A number <= 20000 can have at most 6 distinct prime factors (2*3*5*7*11*13=30030 > 20000? no, 30030>20000, so at most 5? 2*3*5*7*11=2310, *13=30030>20000, so at most 5 distinct small primes, but could have a larger one too. Actually 2*3*5*7*11=2310, 2310*13=30030>20000. So max 5 distinct from small primes, plus possibly one larger. Let's just use a dict mapping n->frozenset.)

def _build_factor_sets():
    """Map each n in [2, _MAX_SUM] to a frozenset of its distinct prime factors."""
    fac = {}
    for n in range(2, _MAX_SUM + 1):
        x = n
        ps = set()
        for p in _SMALL_PRIMES:
            if p * p > x:
                break
            if x % p == 0:
                ps.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            ps.add(x)
        fac[n] = frozenset(ps)
    return fac

_factor_sets = _build_factor_sets()

# Also build a bitmask representation for speed: map each prime to a bit index.
# Collect all primes that appear.
_all_primes = set()
for fs in _factor_sets.values():
    _all_primes |= fs
_all_primes = sorted(_all_primes)
_prime_to_bit = {p: i for i, p in enumerate(_all_primes)}

def _factor_mask(n):
    """Bitmask of distinct prime factors of n."""
    m = 0
    for p in _factor_sets[n]:
        m |= (1 << _prime_to_bit[p])
    return m

_mask_table = [0] * (_MAX_SUM + 1)
for n in range(2, _MAX_SUM + 1):
    _mask_table[n] = _factor_mask(n)

# ---------- scoring ----------
def _score(nums):
    """Number of distinct primes dividing pairwise sums of distinct-index pairs."""
    m = 0
    n = len(nums)
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            m |= _mask_table[ai + nums[j]]
    return bin(m).count("1")

def _compute_other_masks(nums):
    """For each index i, compute the bitmask of all pairwise sums NOT involving i.
    Returns a list of length n where other_masks[i] is the OR of masks for all
    pairs (a,b) with a < b, a != i, b != i."""
    n = len(nums)
    other_masks = [0] * n
    # For each pair (a,b), its mask contributes to other_masks[i] for all i != a and i != b
    for a in range(n):
        for b in range(a + 1, n):
            mask = _mask_table[nums[a] + nums[b]]
            for i in range(n):
                if i != a and i != b:
                    other_masks[i] |= mask
    return other_masks

# ---------- diverse seed generators ----------
def _seed_multiples_of(k, count=N, max_val=MAX_VAL):
    """First `count` distinct multiples of k up to max_val."""
    res = []
    x = k
    while len(res) < count and x <= max_val:
        res.append(x)
        x += k
    return res if len(res) == count else None

def _seed_powers_of_two(count=N, max_val=MAX_VAL):
    res = []
    x = 1
    while len(res) < count and x <= max_val:
        res.append(x)
        x *= 2
    return res if len(res) == count else None

def _seed_arithmetic(start, step, count=N, max_val=MAX_VAL):
    res = []
    x = start
    while len(res) < count and x <= max_val:
        res.append(x)
        x += step
    return res if len(res) == count else None

def _seed_random(rng, count=N, max_val=MAX_VAL):
    return sorted(rng.sample(range(1, max_val + 1), count))

# ---------- local search ----------
def _local_search(nums, budget_iters=20000, tight=False):
    """Hill-climb single-element changes using incremental delta scoring."""
    current = sorted(nums)
    current_score = _score(current)
    iters = 0
    improved = True
    rng = random.Random(42)
    # Precompute a set of "good" candidate values: multiples of highly composite numbers
    # and numbers with few prime factors
    good_candidates = set()
    for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 24, 30, 40, 48, 60, 72, 84, 120, 168, 240, 336, 504, 840):
        x = k
        while x <= MAX_VAL:
            good_candidates.add(x)
            x += k
    # Add numbers that are powers of 2, 3, 5
    for base in (2, 3, 5):
        x = base
        while x <= MAX_VAL:
            good_candidates.add(x)
            x *= base
    good_candidates_list = sorted(good_candidates)
    import bisect

    while improved and iters < budget_iters and not _deadline_hit():
        improved = False
        iters += 1
        n = len(current)
        current_set = set(current)
        
        # Compute other_masks once per iteration
        other_masks = _compute_other_masks(current)
        
        # Try single-element replacement
        for idx in range(n):
            old = current[idx]
            best_new = None
            best_new_score = current_score
            # Try a set of candidate values: neighborhood + structured + random
            candidates = set()
            # Neighborhood size depends on tight mode
            nb = 5 if tight else 20
            for d in range(-nb, nb + 1):
                v = old + d
                if 1 <= v <= MAX_VAL:
                    candidates.add(v)
            # Nearby good candidates
            pos = bisect.bisect_left(good_candidates_list, old)
            gc_span = 3 if tight else 5
            for i in range(max(0, pos - gc_span), min(len(good_candidates_list), pos + gc_span + 1)):
                candidates.add(good_candidates_list[i])
            # A few random values
            n_rand = 2 if tight else 5
            for _ in range(n_rand):
                v = rng.randint(1, MAX_VAL)
                candidates.add(v)

            for v in candidates:
                if v == old or v in current_set:
                    continue
                # Fast delta: OR of (other_masks[idx] | masks for new_val paired with all others)
                m = other_masks[idx]
                for j in range(n):
                    if j != idx:
                        m |= _mask_table[v + current[j]]
                ns = bin(m).count("1")
                if ns < best_new_score:
                    best_new_score = ns
                    best_new = (idx, v)
            if best_new is not None:
                idx, v = best_new
                current[idx] = v
                current = sorted(current)
                current_score = best_new_score
                improved = True
                break
    
    return current, current_score

# ---------- main ----------
def construct():
    rng = random.Random(987654321)

    best_result = None
    best_score = float("inf")

    # Generate diverse seeds
    seeds = []
    # Focus on multiples of highly composite numbers for better factor sharing
    for k in (6, 12, 24, 30, 40, 48, 60, 72, 84, 120, 168, 240, 336, 504, 840):
        s = _seed_multiples_of(k)
        if s:
            seeds.append(s)
    s = _seed_powers_of_two()
    if s:
        seeds.append(s)
    # Arithmetic progressions with steps that are multiples of 6
    for start in (6, 12, 18, 24, 30, 60, 120, 240):
        for step in (6, 12, 24, 30, 48, 60, 120):
            s = _seed_arithmetic(start, step)
            if s:
                seeds.append(s)
    # Random seeds biased toward multiples of 6
    for _ in range(15):
        candidates = [x for x in range(1, MAX_VAL + 1) if x % 6 == 0]
        seeds.append(sorted(rng.sample(candidates, N)))
    # Mix of multiples of 2 and 3
    for _ in range(10):
        candidates = [x for x in range(1, MAX_VAL + 1) if x % 2 == 0 or x % 3 == 0]
        seeds.append(sorted(rng.sample(candidates, N)))

    # Deduplicate seeds (by tuple)
    seen = set()
    unique_seeds = []
    for s in seeds:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            unique_seeds.append(list(s))

    # Evaluate each seed and run local search
    for seed in unique_seeds:
        if _deadline_hit():
            break
        if len(set(seed)) != N or not all(1 <= x <= MAX_VAL for x in seed):
            continue
        try:
            sc = _score(seed)
            if sc < best_score:
                best_score = sc
                best_result = seed[:]
            # Run local search from this seed with adaptive budget
            if _elapsed() > _DEADLINE_SECONDS * 0.7:
                budget = 2000
            else:
                budget = 8000
            improved_list, sc2 = _local_search(sorted(seed), budget_iters=budget)
            if sc2 < best_score:
                best_score = sc2
                best_result = improved_list
        except Exception:
            continue

    # Final safety local-search on the best result
    if best_result is not None and not _deadline_hit():
        cur = best_result[:]
        cur_score = _score(cur)
        if cur_score < best_score:
            best_score = cur_score
        # Use remaining time for final polish with tight neighborhood
        remaining = _DEADLINE_SECONDS - _elapsed()
        if remaining > 5:
            budget = int(min(50000, remaining * 1000))
            improved_list, sc2 = _local_search(sorted(cur), budget_iters=budget, tight=True)
            if sc2 < best_score:
                best_score = sc2
                best_result = improved_list
        # Second polish pass if time remains
        remaining = _DEADLINE_SECONDS - _elapsed()
        if remaining > 3 and best_result is not None:
            budget = int(min(30000, remaining * 800))
            improved_list, sc2 = _local_search(sorted(best_result), budget_iters=budget, tight=True)
            if sc2 < best_score:
                best_score = sc2
                best_result = improved_list

    # Validation
    assert best_result is not None, "no valid result found"
    assert len(set(best_result)) == N, "must have 12 distinct integers"
    assert all(1 <= x <= MAX_VAL for x in best_result), "must be <= 10000 and >= 1"

    return best_result
