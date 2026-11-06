import time
import math

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _compute_powerful_parts_sieve(limit, spf):
    """Compute powerful parts using sieve-like approach."""
    pp = [1] * (limit + 1)
    
    # Get list of primes
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    
    for p in primes:
        pe = p * p  # p^2
        while pe <= limit:
            pe_next = pe * p
            for m in range(pe, limit + 1, pe):
                if m % pe_next != 0:
                    pp[m] *= pe
            pe = pe_next
    
    return pp

def construct():
    start_time = time.time()
    deadline = 55  # safety margin before 60s
    
    LO = 10000
    HI = 1000000
    
    # Build SPF up to HI
    spf = _build_spf(HI)
    
    # Check time
    if time.time() - start_time > deadline:
        return LO
    
    # Compute powerful parts for all numbers in [0, HI]
    pp = _compute_powerful_parts_sieve(HI, spf)
    
    if time.time() - start_time > deadline:
        return LO
    
    # Find n in [LO, HI-1] that maximizes pp[n] * pp[n+1]
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    
    for n in range(LO, HI):
        if time.time() - start_time > deadline:
            break
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
