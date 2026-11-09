"""Rewritten program: search for a large set of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 55.0  # safety margin before 60s kill

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    f = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f

def _build_prime_table():
    """Generate primes up to 40 and compute exponent vectors for 2..40."""
    primes = []
    for n in range(2, 41):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    
    all_primes_set = set()
    factorizations = {}
    for n in range(2, 41):
        f = _factorize(n, primes)
        factorizations[n] = f
        all_primes_set.update(f.keys())
    
    prime_list = sorted(all_primes_set)
    prime_index = {p: i for i, p in enumerate(prime_list)}
    dim = len(prime_list)
    
    exp_vecs = {}
    for n in range(2, 41):
        v = [0] * dim
        for p, e in factorizations[n].items():
            v[prime_index[p]] = e
        exp_vecs[n] = tuple(v)
    
    return primes, exp_vecs, dim

def _subset_products_distinct_fast(nums, exp_vecs, dim):
    """Check if all 2^k subset products of nums are distinct using exponent vectors."""
    if not nums:
        return True
    
    sums = {tuple([0]*dim)}
    for num in nums:
        v = exp_vecs[num]
        new_sums = set()
        for s in sums:
            new_sums.add(tuple(s[j] + v[j] for j in range(dim)))
        sums.update(new_sums)
        if len(sums) != (1 << len(nums)):
            return False
    return True

def construct():
    start_time = time.time()
    
    primes, exp_vecs, dim = _build_prime_table()
    candidates = list(range(2, 41))
    
    def num_prime_factors(n):
        f = _factorize(n, primes)
        return sum(f.values())
    
    best_set = []
    best_size = 0
    
    # Deterministic orderings
    orderings = [
        sorted(candidates, key=lambda x: (0 if x in primes else 1, num_prime_factors(x), x)),
        sorted(candidates, key=lambda x: (num_prime_factors(x), x)),
        candidates,
        list(reversed(candidates)),
        sorted(candidates, key=lambda x: (0 if x in primes else 1, x)),
    ]
    
    for ordering in orderings:
        if time.time() - start_time > DEADLINE:
            break
        current = []
        for num in ordering:
            if time.time() - start_time > DEADLINE:
                break
            test = current + [num]
            if _subset_products_distinct_fast(test, exp_vecs, dim):
                current = test
        if len(current) > best_size:
            best_set = list(current)
            best_size = len(current)
    
    # Random restarts with shuffled orderings
    rng = random.Random(42)
    while time.time() - start_time < DEADLINE:
        ordering = list(candidates)
        rng.shuffle(ordering)
        current = []
        for num in ordering:
            if time.time() - start_time > DEADLINE:
                break
            test = current + [num]
            if _subset_products_distinct_fast(test, exp_vecs, dim):
                current = test
        if len(current) > best_size:
            best_set = list(current)
            best_size = len(current)
    
    # Local search on best_set
    if time.time() - start_time < DEADLINE:
        improved = True
        while improved and (time.time() - start_time < DEADLINE):
            improved = False
            # Try adding one element
            for num in candidates:
                if num in best_set:
                    continue
                if time.time() - start_time > DEADLINE:
                    break
                test = best_set + [num]
                if _subset_products_distinct_fast(test, exp_vecs, dim):
                    best_set = test
                    improved = True
                    break
            # Try swapping: remove one, add another
            if time.time() - start_time > DEADLINE:
                break
            for i in range(len(best_set)):
                if time.time() - start_time > DEADLINE:
                    break
                reduced = best_set[:i] + best_set[i+1:]
                for num in candidates:
                    if num in reduced:
                        continue
                    if time.time() - start_time > DEADLINE:
                        break
                    test = reduced + [num]
                    if len(test) > len(best_set):
                        if _subset_products_distinct_fast(test, exp_vecs, dim):
                            best_set = test
                            improved = True
                            break
                if improved:
                    break
    
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
