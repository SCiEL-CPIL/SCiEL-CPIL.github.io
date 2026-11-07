"""Rewritten program: model subset-product distinctness as integer linear independence
of prime-exponent vectors; use exact rational Gaussian elimination + greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _factorize(n):
    """Return dict {prime: exponent} for n."""
    d = {}
    temp = n
    f = 2
    while f * f <= temp:
        while temp % f == 0:
            d[f] = d.get(f, 0) + 1
            temp //= f
        f += 1
    if temp > 1:
        d[temp] = d.get(temp, 0) + 1
    return d

def _build_vectors(nums):
    """Build exponent vectors for each number, return (vectors, primes)."""
    prime_set = set()
    for n in nums:
        for p in _factorize(n):
            prime_set.add(p)
    primes = sorted(prime_set)
    idx = {p: i for i, p in enumerate(primes)}
    vecs = []
    for n in nums:
        f = _factorize(n)
        v = [0] * len(primes)
        for p, e in f.items():
            v[idx[p]] = e
        vecs.append(v)
    return vecs, primes

def _is_independent(vecs):
    """
    Check if the list of integer vectors (rows) are linearly independent over Q.
    Use Gaussian elimination with exact rationals.
    Returns True if independent, False if a dependency exists.
    """
    if not vecs:
        return True
    # Work on a copy as list of lists of Fraction
    m = len(vecs)
    n = len(vecs[0])
    mat = [[Fraction(x) for x in v] for v in vecs]
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for r in range(rank, m):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        # Normalize pivot row
        pv = mat[rank][col]
        for j in range(col, n):
            mat[rank][j] /= pv
        # Eliminate
        for r in range(m):
            if r == rank:
                continue
            factor = mat[r][col]
            if factor != 0:
                for j in range(col, n):
                    mat[r][j] -= factor * mat[rank][j]
        rank += 1
        if rank == m:
            return True
    return rank == m

def _can_add(vecs, new_vec):
    """Check if adding new_vec keeps the set independent."""
    test = vecs + [new_vec]
    return _is_independent(test)

def _greedy_build(candidates, vecs_dict, order):
    """Greedy: add in order if independent."""
    selected = []
    selected_vecs = []
    for c in order:
        v = vecs_dict[c]
        if _can_add(selected_vecs, v):
            selected.append(c)
            selected_vecs.append(v)
    return selected

def _local_search(selected, all_nums, vecs_dict, time_limit):
    """Try to add elements not in selected, in random orders."""
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        # Try adding each candidate not in selected
        for c in all_nums:
            if c in selected:
                continue
            v = vecs_dict[c]
            if _can_add([vecs_dict[x] for x in selected], v):
                selected.append(c)
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    all_nums = list(range(2, 41))
    
    # Precompute vectors for all candidates
    vecs_dict = {}
    for n in all_nums:
        vecs_dict[n] = _factorize(n)
        # Convert to dense vector over all primes that appear
    # Get all primes
    all_primes = set()
    for n in all_nums:
        for p in vecs_dict[n]:
            all_primes.add(p)
    all_primes = sorted(all_primes)
    pidx = {p: i for i, p in enumerate(all_primes)}
    
    # Convert to dense vectors
    dense_vecs = {}
    for n in all_nums:
        f = vecs_dict[n]
        v = [0] * len(all_primes)
        for p, e in f.items():
            v[pidx[p]] = e
        dense_vecs[n] = v
    
    # Try multiple orderings
    orderings = []
    
    # Order 1: primes first (1-dimensional vectors, most independent)
    primes = [n for n in all_nums if len(_factorize(n)) == 1 and _factorize(n)[list(_factorize(n).keys())[0]] == 1]
    # Actually, primes are numbers with exactly one prime factor with exponent 1
    primes = [n for n in all_nums if sum(_factorize(n).values()) == 1]
    orderings.append(primes + [x for x in all_nums if x not in primes])
    
    # Order 2: by number of prime factors (ascending) - simpler vectors first
    def _pf_count(n):
        return sum(_factorize(n).values())
    orderings.append(sorted(all_nums, key=_pf_count))
    
    # Order 3: by largest prime factor
    def _lpf(n):
        return max(_factorize(n).keys())
    orderings.append(sorted(all_nums, key=_lpf))
    
    # Order 4: ascending
    orderings.append(all_nums[:])
    
    # Order 5: descending
    orderings.append(sorted(all_nums, reverse=True))
    
    # Order 6: by sum of exponents
    def _sum_exp(n):
        return sum(_factorize(n).values())
    orderings.append(sorted(all_nums, key=lambda x: (sum(_factorize(x).values()), x)))
    
    best = []
    for order in orderings:
        if time.time() > deadline:
            break
        result = _greedy_build(all_nums, dense_vecs, order)
        if len(result) > len(best):
            best = result
            if len(best) >= 18:
                break
    
    # Random restarts
    rng = random.Random(12345)
    for it in range(500):
        if time.time() > deadline:
            break
        random_order = all_nums[:]
        rng.shuffle(random_order)
        result = _greedy_build(all_nums, dense_vecs, random_order)
        if len(result) > len(best):
            best = result
            if len(best) >= 18:
                break
    
    # Local search: try to add more
    if len(best) < 18:
        ls_start = time.time()
        ls_deadline = min(deadline, ls_start + 20.0)
        while (time.time() < ls_deadline) and len(best) < 18:
            added = False
            # Try each candidate not in best
            for c in all_nums:
                if c in best:
                    continue
                if _can_add([dense_vecs[x] for x in best], dense_vecs[c]):
                    best.append(c)
                    added = True
                    break
            if not added:
                break
    
    # Final verification using exact product check
    def _verify(nums):
        k = len(nums)
        products = set()
        products.add(1)
        for i in range(k):
            new_prods = []
            for p in products:
                np = p * nums[i]
                if np in products:
                    return False
                new_prods.append(np)
            products.update(new_prods)
        return True
    
    if not _verify(best):
        # Fallback: just use primes
        best = primes
    
    # Ensure types
    result = []
    seen = set()
    for x in best:
        if isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40 and x not in seen:
            seen.add(x)
            result.append(x)
    
    return result
# EVOLVE-BLOCK-END
