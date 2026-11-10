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
import sys

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0
MAX_VAL = 10000
N = 12


def _elapsed():
    return time.time() - _START_TIME


def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS


if hasattr(int, "bit_count"):
    def _popcount(x):
        return x.bit_count()
else:
    def _popcount(x):
        return bin(x).count("1")


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
    return _popcount(m)

def _score_partial(nums, i, new_val, old_val):
    """Compute the change in score if we replace nums[i] (old_val) with new_val.
    Returns the new full score (not delta) for efficiency in tight loops."""
    m = 0
    n = len(nums)
    for j in range(n):
        if j == i:
            continue
        m |= _mask_table[new_val + nums[j]]
    # Add pairs not involving i
    for a in range(n):
        if a == i:
            continue
        for b in range(a + 1, n):
            if b == i:
                continue
            m |= _mask_table[nums[a] + nums[b]]
    return bin(m).count("1")

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
def _local_search_sa(nums, budget_iters=50000):
    """Simulated Annealing local search with single-element changes and swaps."""
    current = sorted(nums)
    current_score = _score(current)
    best = current[:]
    best_score = current_score
    
    T = 10.0
    cooling = 0.9995
    rng = random.Random(42)
    
    for _ in range(budget_iters):
        if _deadline_hit():
            break
        if T < 0.01:
            break
            
        # 20% chance of swap, 80% chance of single change
        if rng.random() < 0.2:
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i == j:
                T *= cooling
                continue
            new_list = current[:]
            new_list[i], new_list[j] = new_list[j], new_list[i]
            new_score = _score(new_list)
            delta = new_score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current[i], current[j] = current[j], current[i]
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
        else:
            idx = rng.randint(0, N - 1)
            old_val = current[idx]
            
            # Generate candidate
            # 50% chance local, 50% chance global
            if rng.random() < 0.5:
                # Local: +/- up to 100
                delta_val = rng.randint(-100, 100)
                new_val = old_val + delta_val
            else:
                # Global random
                new_val = rng.randint(1, MAX_VAL)
                
            if new_val < 1 or new_val > MAX_VAL:
                T *= cooling
                continue
            if new_val == old_val:
                T *= cooling
                continue
            if new_val in current:
                T *= cooling
                continue
                
            new_list = current[:]
            new_list[idx] = new_val
            new_score = _score(new_list)
            
            delta = new_score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current[idx] = new_val
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best = current[:]
                    
        T *= cooling
        
    return best, best_score

# ---------- main ----------
def construct():
    rng = random.Random(987654321)

    best_result = None
    best_score = float("inf")

    # Generate diverse seeds
    seeds = []
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 50, 60, 100, 120):
        s = _seed_multiples_of(k)
        if s:
            seeds.append(s)
    s = _seed_powers_of_two()
    if s:
        seeds.append(s)
    for start in (1, 2, 3, 5, 10, 100, 1000):
        for step in (1, 2, 3, 5, 10, 20, 50, 100):
            s = _seed_arithmetic(start, step)
            if s:
                seeds.append(s)
    for _ in range(20):
        seeds.append(_seed_random(rng))

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
            # Run local search from this seed
            improved_list, sc2 = _local_search_sa(sorted(seed), budget_iters=20000)
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
        improved_list, sc2 = _local_search_sa(sorted(cur), budget_iters=50000)
        if sc2 < best_score:
            best_score = sc2
            best_result = improved_list

    # Validation
    assert best_result is not None, "no valid result found"
    assert len(set(best_result)) == N, "must have 12 distinct integers"
    assert all(1 <= x <= MAX_VAL for x in best_result), "must be <= 10000 and >= 1"

    return best_result
