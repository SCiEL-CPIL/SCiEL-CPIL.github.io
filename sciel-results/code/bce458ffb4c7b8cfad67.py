"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# Hardcoded 16-element set with distinct subset products.
# From prime 2: {2, 4, 16} (exponents 1,2,4 — subset sums {0,1,2,3,4,5,6} all distinct)
# From prime 3: {3, 9} (exponents 1,2)
# From prime 5: {5, 25} (exponents 1,2)
# From primes 7,11,13,17,19,23,29,31,37: one each (exponent 1)
_BASE = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]


def _verify(set_):
    """Verify that all 2^k subset products are distinct. Uses prime-exponent vectors."""
    # Factor each number into prime exponent vector
    primes = []
    for n in range(2, 41):
        if _is_prime(n):
            primes.append(n)
    
    def factor(n):
        v = {}
        for p in primes:
            while n % p == 0:
                v[p] = v.get(p, 0) + 1
                n //= p
        return tuple(v.get(p, 0) for p in primes)
    
    vecs = [factor(a) for a in set_]
    k = len(set_)
    
    # Enumerate all subset product exponent vectors
    products = set()
    for mask in range(1 << k):
        v = [0] * len(primes)
        for i in range(k):
            if mask & (1 << i):
                for j in range(len(primes)):
                    v[j] += vecs[i][j]
        products.add(tuple(v))
    
    return len(products) == (1 << k)


def _try_extend(base, remaining, current_size, target):
    """Try to extend base by picking from remaining to reach target size."""
    if current_size >= target:
        return base
    
    for i in range(len(remaining)):
        candidate = remaining[i]
        new_set = base + [candidate]
        if _verify(new_set):
            rest = remaining[:i] + remaining[i+1:]
            return _try_extend(new_set, rest, current_size + 1, target)
    return base


def construct():
    import time
    t0 = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    # Start with the known 16-element solution
    result = list(_BASE)
    
    if not _verify(result):
        # Fallback: greedy construction
        result = []
        for a in range(2, 41):
            trial = result + [a]
            if len(trial) <= 18 and _verify(trial):
                result = trial
        return tuple(result)
    
    # Try to reach 17 or 18 by adding more numbers
    remaining = [a for a in range(2, 41) if a not in result]
    
    for target in [18, 17]:
        if time.time() - t0 > deadline:
            break
        candidate = _try_extend(result, remaining, len(result), target)
        if len(candidate) > len(result):
            result = candidate
            remaining = [a for a in range(2, 41) if a not in result]
    
    return tuple(result)
# EVOLVE-BLOCK-END
