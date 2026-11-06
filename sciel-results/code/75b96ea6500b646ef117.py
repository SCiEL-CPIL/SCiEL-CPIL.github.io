"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE = 75.0  # seconds, safety margin before 90s kill

def _sieve_spf(n):
    """Smallest-prime-factor sieve up to n (inclusive)."""
    spf = np.zeros(n + 1, dtype=np.int64)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i::i] = i
    return spf

def _powerful_part_from_spf(x, spf):
    """Compute powerful part of x using SPF table. Returns 1 if no prime power with e>=2."""
    if x < 4:
        return 1
    pp = 1
    while x > 1:
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def _precompute_powerful_parts(N, spf):
    """
    Precompute powerful part for every integer in [1, N] using a segmented
    sieve approach: for each prime p, for each power p^e with e>=2, multiply
    p^e into all multiples of p^e that are NOT multiples of p^(e+1).
    """
    pp = np.ones(N + 1, dtype=np.int64)
    
    # Find all primes up to N
    primes = []
    for i in range(2, N + 1):
        if spf[i] == i:  # i is prime
            primes.append(i)
    
    for p in primes:
        # p^2 is the minimum power with e>=2
        p2 = p * p
        if p2 > N:
            break
        # For e=2,3,...: multiply p^e into numbers divisible by p^e but not p^(e+1)
        pe = p2
        while pe <= N:
            pe_next = pe * p
            # Numbers divisible by pe but not by pe_next
            # These are multiples of pe: pe, 2*pe, 3*pe, ...
            # but exclude those also divisible by pe_next
            # i.e., multiples of pe where (k*pe) % pe_next != 0
            # k*pe % (pe*p) != 0  =>  k % p != 0
            for k in range(1, N // pe + 1):
                if k % p != 0:
                    m = k * pe
                    pp[m] *= pe
            pe = pe_next
    
    return pp

def _precompute_powerful_parts_fast(N, spf):
    """
    Segmented sieve: for each prime p, for each power p^e with e>=2, multiply
    p^e into all multiples of p^e that are NOT multiples of p^(e+1).
    This gives O(N log log N) total work instead of O(N log N).
    """
    pp = np.ones(N + 1, dtype=np.int64)
    
    # Find all primes up to sqrt(N) (primes p where p^2 <= N)
    limit = int(math.isqrt(N))
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    
    for p in primes:
        p2 = p * p
        pe = p2
        while pe <= N:
            pe_next = pe * p
            # For each multiple m of pe where m is NOT a multiple of pe_next:
            # m = k * pe, and k % p != 0
            # Vectorized: create mask for k in [1, N//pe] where k % p != 0
            k_max = N // pe
            if k_max >= 1:
                ks = np.arange(1, k_max + 1)
                mask = (ks % p) != 0
                if np.any(mask):
                    ms = ks[mask] * pe
                    pp[ms] *= pe
            pe = pe_next
    
    return pp

def construct():
    t0 = time.time()
    N = 1000000
    LO = 10000
    
    # Build SPF
    spf = _sieve_spf(N)
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 5:
        return LO
    
    # Precompute powerful parts
    pp = _precompute_powerful_parts_fast(N, spf)
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 3:
        return LO
    
    # Sanity check: verify a few values
    def _verify_pp(x, spf):
        """Independent check of powerful part."""
        if x < 4:
            return 1
        y = x
        val = 1
        while y > 1:
            p = spf[y]
            e = 0
            while y % p == 0:
                y //= p
                e += 1
            if e >= 2:
                val *= p ** e
        return val
    
    for test_x in [10000, 10001, 50000, 999999, 1000000]:
        expected = _verify_pp(test_x, spf)
        if pp[test_x] != expected:
            # Fall back to per-number factorization if sieve is wrong
            pp = np.ones(N + 1, dtype=np.int64)
            for x in range(4, N + 1):
                y = x
                val = 1
                while y > 1:
                    p = spf[y]
                    e = 0
                    while y % p == 0:
                        y //= p
                        e += 1
                    if e >= 2:
                        val *= p ** e
                pp[x] = val
            break
    
    # Find max of pp[n] * pp[n+1] for n in [LO, N-1]
    products = pp[LO:N] * pp[LO+1:N+1]
    best_idx = np.argmax(products)
    best_n = LO + best_idx
    best_val = products[best_idx]
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 2:
        return int(best_n)
    
    # Verify and return
    return int(best_n)
# EVOLVE-BLOCK-END
