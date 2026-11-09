import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 210
    max_val = 10210

    # Factorize 210: 2^1 * 3^1 * 5^1 * 7^1
    # Target: XOR of selected parity vectors must equal parity of 210
    # Parity of 210: bits for primes 2,3,5,7 are set

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210
    target = masks[210]

    # We want to find the minimum M such that there exists a subset of {211, ..., M}
    # whose XOR of masks equals target, with at most 100 elements.
    # 
    # Strategy: Use Gaussian elimination to find the minimum M where target is in the span,
    # then try to find a sparse solution (few elements) within that range.
    #
    # First, find the minimum M where target is in the span.
    # Then, use a second phase to find a sparse solution.

    # Phase 1: Find minimum M where target is in span
    basis = {}
    min_M = None

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # history bit for this candidate

        # Insert v into basis with history h
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                basis[p] = (tv, th)
                break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                break

        if tv == 0:
            min_M = M
            break

    if min_M is None:
        # Fallback
        return [840]

    # Phase 2: Find a sparse solution within {211, ..., min_M}
    # Use a greedy approach: try to find a small subset whose XOR equals target.
    # We can use the basis to express target, but the solution may have many elements.
    # Instead, try a different approach: use the basis to find all solutions and pick the sparsest.
    # 
    # Actually, let's use a different strategy:
    # 1. Build the basis for {211, ..., min_M}
    # 2. Find one solution via the basis
    # 3. Try to reduce the number of elements by XOR-ing with elements in the kernel

    # Rebuild basis for {211, ..., min_M}
    basis2 = {}
    for M in range(211, min_M + 1):
        v = masks[M]
        h = 1 << (M - 211)
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis2:
                bv, bh = basis2[p]
                tv ^= bv
                th ^= bh
            else:
                basis2[p] = (tv, th)
                break

    # Find one solution
    tv = target
    th = 0
    while tv:
        p = tv.bit_length() - 1
        if p in basis2:
            bv, bh = basis2[p]
            tv ^= bv
            th ^= bh
        else:
            break

    if tv != 0:
        # Should not happen
        return [840]

    # Extract solution
    sol_mask = th
    sol = []
    hh = sol_mask
    while hh:
        lsb = hh & -hh
        idx = lsb.bit_length() - 1
        sol.append(211 + idx)
        hh ^= lsb

    if len(sol) <= 100:
        # Verify
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        if s * s == prod:
            return [int(x) for x in sol]

    # Try to reduce the solution by XOR-ing with kernel elements
    # The kernel consists of all vectors in {211, ..., min_M} that are in the span
    # but not used in the basis.
    # 
    # Alternative: try a different approach. Use a SAT-like search for small subsets.
    # Since min_M is likely small, we can try to find a small subset.
    
    # Try to find a subset of size 1, 2, 3, ... up to some limit
    # For size 1: check if any single element has mask == target
    # For size 2: check if any pair has XOR == target
    # etc.

    # First, check size 1
    for M in range(211, min_M + 1):
        if masks[M] == target:
            prod = 210 * M
            s = math.isqrt(prod)
            if s * s == prod:
                return [int(M)]

    # Check size 2
    for i in range(211, min_M + 1):
        for j in range(i + 1, min_M + 1):
            if masks[i] ^ masks[j] == target:
                prod = 210 * i * j
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(i), int(j)]

    # Check size 3
    for i in range(211, min_M + 1):
        for j in range(i + 1, min_M + 1):
            for k in range(j + 1, min_M + 1):
                if masks[i] ^ masks[j] ^ masks[k] == target:
                    prod = 210 * i * j * k
                    s = math.isqrt(prod)
                    if s * s == prod:
                        return [int(i), int(j), int(k)]

    # If we get here, the solution from the basis has > 100 elements
    # and we couldn't find a small subset. Return the basis solution if valid.
    if len(sol) <= 100:
        return [int(x) for x in sol]

    # Fallback: return a known valid solution
    return [840]
