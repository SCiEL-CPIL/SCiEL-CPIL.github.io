import time

def _factorize(n, primes):
    """Return exponent vector of n over given primes."""
    vec = []
    for p in primes:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        vec.append(e)
    if n != 1:
        raise ValueError(f"{n} has prime factor not in list")
    return tuple(vec)

def _check_distinct_vectors(vectors):
    """Check all 2^k subset sums of vectors are distinct."""
    k = len(vectors)
    dim = len(vectors[0])
    sums = set()
    # Use iterative subset sum enumeration
    sums.add(tuple([0] * dim))
    for i in range(k):
        vi = vectors[i]
        # For each existing sum, add vi to create new sums
        new_sums = []
        for s in list(sums):
            ns = tuple(s[j] + vi[j] for j in range(dim))
            if ns in sums:
                return False
            new_sums.append(ns)
        sums.update(new_sums)
    return True

def construct():
    """
    Returns distinct integers in [2, 40] such that all 2^k subset products are distinct.
    Uses prime exponent vector representation for efficient distinctness checking.
    Starts from a known 16-element construction and attempts to add more elements.
    """
    t0 = time.time()
    deadline = 75.0  # safety margin before 90s limit

    # All primes in [2, 40]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Build exponent vectors for all numbers in [2, 40]
    valid_numbers = {}
    for n in range(2, 41):
        try:
            vec = _factorize(n, primes)
            valid_numbers[n] = vec
        except ValueError:
            pass

    # Base construction: prime powers with superincreasing exponents
    base_nums = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    base_vecs = [valid_numbers[n] for n in base_nums]

    assert _check_distinct_vectors(base_vecs), "Base invalid"
    best_vecs = list(base_vecs)
    best_nums = list(base_nums)

    # Try to add one more number to get 17
    base_set = set(base_nums)
    candidates = [n for n in range(2, 41) if n not in base_set]

    # Sort by number of distinct prime factors (simpler first)
    def _pf_count(n):
        c = 0
        for p in primes:
            if n % p == 0:
                c += 1
        return c

    candidates.sort(key=_pf_count)

    for cand in candidates:
        if time.time() > deadline:
            break
        test_vecs = base_vecs + [valid_numbers[cand]]
        if _check_distinct_vectors(test_vecs):
            best_vecs = test_vecs
            best_nums = base_nums + [cand]
            break

    # If we have 17, try for 18
    if len(best_nums) == 17:
        best_set = set(best_nums)
        candidates2 = [n for n in range(2, 41) if n not in best_set]
        candidates2.sort(key=_pf_count)
        for cand in candidates2:
            if time.time() > deadline:
                break
            test_vecs = best_vecs + [valid_numbers[cand]]
            if _check_distinct_vectors(test_vecs):
                best_vecs = test_vecs
                best_nums = best_nums + [cand]
                break

    return list(best_nums)
